use std::cell::Cell;
use std::collections::LinkedList;
//use std::time::Instant;

use crate::cs::CsMatrix;
use crate::ops::serial::{OperationError, OperationErrorKind};
use crate::ops::Op;
use crate::pattern::{SparsityPattern, SparsityPatternFormatError};
use crate::SparseEntryMut;
use fxhash::FxHashMap;
use itertools::Itertools;
use nalgebra::{ClosedAdd, ClosedMul, DMatrixSlice, DMatrixSliceMut, Scalar};
use num_traits::{One, Zero};
use rayon::prelude::*;
use thread_local::ThreadLocal;

/// Helper functionality for implementing CSR/CSC SPMM.
///
/// Since CSR/CSC matrices are basically transpositions of each other, which lets us use the same
/// algorithm for the SPMM implementation. The implementation here is written in a CSR-centric
/// manner. This means that when using it for CSC, the order of the matrices needs to be
/// reversed (since transpose(AB) = transpose(B) * transpose(A) and CSC(A) = transpose(CSR(A)).
///
/// We assume here that the matrices have already been verified to be dimensionally compatible.
pub fn spmm_cs_prealloc<T>(
    beta: T,
    c: &mut CsMatrix<T>,
    alpha: T,
    a: &CsMatrix<T>,
    b: &CsMatrix<T>,
) -> Result<(), OperationError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    let some_val = Zero::zero();
    let mut scratchpad_values: Vec<T> = vec![some_val; b.pattern().minor_dim()];
    for i in 0..c.pattern().major_dim() {
        let a_lane_i = a.get_lane(i).unwrap();

        let mut c_lane_i = c.get_lane_mut(i).unwrap();

        for (&k, a_ik) in a_lane_i.minor_indices().iter().zip(a_lane_i.values()) {
            let b_lane_k = b.get_lane(k).unwrap();
            let alpha_aik = alpha.clone() * a_ik.clone();
            for (j, b_kj) in b_lane_k.minor_indices().iter().zip(b_lane_k.values()) {
                // Determine the location in C to append the value
                scratchpad_values[*j] += alpha_aik.clone() * b_kj.clone();
            }
        }
        // sort the indices, and then access the relevant indices (in sorted order) from values
        // into C.
        let (indices, values) = c_lane_i.indices_and_values_mut();

        values
            .iter_mut()
            .zip(indices)
            .for_each(|(output_ref, index)| {
                *output_ref = beta.clone() * output_ref.clone() + scratchpad_values[*index].clone();
                scratchpad_values[*index] = Zero::zero();
            });
    }

    Ok(())
}

struct ThreadLocalResult<T> {
    output_rows: LinkedList<(usize, FxHashMap<usize, T>)>, //this is a list of output rows that the thread has computed till now.
                                                           //first element of each ordered pair in this list tells you which row we're talking about,
                                                           //and the second element is a small hashmap representing the row itself.
}

fn slice_to_subslices<'a, T>(
    some_slice: &'a mut [T],
    cut_points: &Vec<usize>,
) -> LinkedList<&'a mut [T]> {
    cut_points
        .iter()
        .zip(cut_points.iter().skip(1))
        .fold(LinkedList::new(), |mut list, (l, r)| {
            assert!(*l < some_slice.len() && some_slice.len() >= *r);
            let base_ptr = some_slice.as_mut_ptr();
            unsafe {
                let subslice = std::slice::from_raw_parts_mut(base_ptr.offset(*l as isize), r - l);
                list.push_back(subslice);
            }
            list
        })
}

/// each thread has a thread-local struct that it keeps on growing.
/// This contains a non-contiguous list of rows of the output sparse matrix. Which is basically
/// DCSR
pub fn spmm_cs_prealloc_parallel<T>(
    alpha: T,
    a: &CsMatrix<T>,
    b: &CsMatrix<T>,
) -> Result<CsMatrix<T>, SparsityPatternFormatError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One + Send + Sync + Sized + Copy,
{
    let num_rows = a.pattern().major_dim();
    let all_thread_results = ThreadLocal::new();
    (0..num_rows).into_par_iter().for_each(|a_row_id| {
        let my_state = all_thread_results.get_or(|| {
            Cell::new(ThreadLocalResult {
                output_rows: LinkedList::new(),
            })
        });
        let mut old_result = my_state.replace(ThreadLocalResult {
            output_rows: LinkedList::new(),
        });
        let a_lane = a.get_lane(a_row_id).unwrap();
        let _b_lane = b.get_lane(a_row_id);
        let c_row_hash = a_lane.minor_indices().iter().zip(a_lane.values()).fold(
            FxHashMap::default(),
            |mut scatter_values: FxHashMap<usize, T>, a_element_tuple| {
                let (k, a_ik) = a_element_tuple;
                let b_lane = b.get_lane(*k);
                b_lane.map(|b_row| {
                    b_row
                        .minor_indices()
                        .into_iter()
                        .zip(b_row.values().into_iter())
                        .for_each(|(j, b_kj)| {
                            let some_entry = scatter_values.entry(*j).or_insert(Zero::zero());
                            *some_entry += alpha.clone() * a_ik.clone() * b_kj.clone();
                        });
                });
                scatter_values
            },
        );
        //c_row_hash is now one full row of C
        old_result.output_rows.push_back((a_row_id, c_row_hash));
        my_state.replace(old_result);
    });
    let all_results =
        all_thread_results
            .into_iter()
            .fold(LinkedList::new(), |mut list, some_result| {
                let mut some_result = some_result.replace(ThreadLocalResult {
                    output_rows: LinkedList::new(),
                });
                list.append(&mut some_result.output_rows);
                list
            });
    let num_nnzs = all_results
        .iter()
        .map(|(_row_id, row_hash)| row_hash.len())
        .sum();
    let mut indices = vec![0usize; num_nnzs];
    let mut values = vec![Zero::zero(); num_nnzs];

    //Make a list of mutable slices of indices and values.

    let all_results: LinkedList<_> = all_results
        .into_iter()
        .sorted_unstable_by(|(left_id, _), (right_id, _)| left_id.cmp(right_id))
        .collect();
    let offsets = all_results.iter().fold(vec![0], |mut v, (_, row_hash)| {
        v.push(v.last().unwrap_or(&0) + row_hash.len());
        v
    });
    let list_of_index_slices = slice_to_subslices(&mut indices, &offsets);
    let list_of_values_slices = slice_to_subslices(&mut values, &offsets);
    let all_results = all_results
        .into_iter()
        .zip(list_of_index_slices.into_iter())
        .zip(list_of_values_slices.into_iter())
        .map(|(((_id, hash), index_slice), value_slice)| (hash, index_slice, value_slice))
        .collect::<LinkedList<_>>();
    all_results
        .into_par_iter()
        .for_each(|(mut row_hash, index_ref, value_ref)| {
            //let indices_ref = get_mut_slice(&mut indices, offsets[row_id], offsets[row_id + 1]);
            //let values_ref = get_mut_slice(&mut values, offsets[row_id], offsets[row_id + 1]);
            //drain the hashmap of this row into a slice of the indices and the values array
            //first get the column co-ordinates in sorted order for the given row, and then for each
            //non-zero, copy to the above slices at the right position
            row_hash
                .drain()
                .sorted_unstable_by(|l, r| l.0.cmp(&r.0))
                .enumerate()
                .for_each(|(ind, (column_index, nnz))| {
                    index_ref[ind] = column_index;
                    value_ref[ind] = nnz;
                });
        });

    SparsityPattern::try_from_offsets_and_indices(
        num_rows,
        b.pattern().minor_dim(),
        offsets,
        indices,
    )
    .map(|pattern| CsMatrix::from_pattern_and_values(pattern, values))
}

fn spadd_cs_unexpected_entry() -> OperationError {
    OperationError::from_kind_and_message(
        OperationErrorKind::InvalidPattern,
        String::from("Found entry in `op(a)` that is not present in `c`."),
    )
}

/// Helper functionality for implementing CSR/CSC SPADD.
pub fn spadd_cs_prealloc<T>(
    beta: T,
    c: &mut CsMatrix<T>,
    alpha: T,
    a: Op<&CsMatrix<T>>,
) -> Result<(), OperationError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    match a {
        Op::NoOp(a) => {
            for (mut c_lane_i, a_lane_i) in c.lane_iter_mut().zip(a.lane_iter()) {
                if beta != T::one() {
                    for c_ij in c_lane_i.values_mut() {
                        *c_ij *= beta.clone();
                    }
                }

                let (mut c_minors, mut c_vals) = c_lane_i.indices_and_values_mut();
                let (a_minors, a_vals) = (a_lane_i.minor_indices(), a_lane_i.values());

                for (a_col, a_val) in a_minors.iter().zip(a_vals) {
                    // TODO: Use exponential search instead of linear search.
                    // If C has substantially more entries in the row than A, then a line search
                    // will needlessly visit many entries in C.
                    let (c_idx, _) = c_minors
                        .iter()
                        .enumerate()
                        .find(|(_, c_col)| *c_col == a_col)
                        .ok_or_else(spadd_cs_unexpected_entry)?;
                    c_vals[c_idx] += alpha.clone() * a_val.clone();
                    c_minors = &c_minors[c_idx..];
                    c_vals = &mut c_vals[c_idx..];
                }
            }
        }
        Op::Transpose(a) => {
            if beta != T::one() {
                for c_ij in c.values_mut() {
                    *c_ij *= beta.clone();
                }
            }

            for (i, a_lane_i) in a.lane_iter().enumerate() {
                for (&j, a_val) in a_lane_i.minor_indices().iter().zip(a_lane_i.values()) {
                    let a_val = a_val.clone();
                    let alpha = alpha.clone();
                    match c.get_entry_mut(j, i).unwrap() {
                        SparseEntryMut::NonZero(c_ji) => *c_ji += alpha * a_val,
                        SparseEntryMut::Zero => return Err(spadd_cs_unexpected_entry()),
                    }
                }
            }
        }
    }
    Ok(())
}

/// Helper functionality for implementing CSR/CSC SPMM.
///
/// The implementation essentially assumes that `a` is a CSR matrix. To use it with CSC matrices,
/// the transposed operation must be specified for the CSC matrix.
pub fn spmm_cs_dense<T>(
    beta: T,
    mut c: DMatrixSliceMut<'_, T>,
    alpha: T,
    a: Op<&CsMatrix<T>>,
    b: Op<DMatrixSlice<'_, T>>,
) where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    match a {
        Op::NoOp(a) => {
            for j in 0..c.ncols() {
                let mut c_col_j = c.column_mut(j);
                for (c_ij, a_row_i) in c_col_j.iter_mut().zip(a.lane_iter()) {
                    let mut dot_ij = T::zero();
                    for (&k, a_ik) in a_row_i.minor_indices().iter().zip(a_row_i.values()) {
                        let b_contrib = match b {
                            Op::NoOp(ref b) => b.index((k, j)),
                            Op::Transpose(ref b) => b.index((j, k)),
                        };
                        dot_ij += a_ik.clone() * b_contrib.clone();
                    }
                    *c_ij = beta.clone() * c_ij.clone() + alpha.clone() * dot_ij;
                }
            }
        }
        Op::Transpose(a) => {
            // In this case, we have to pre-multiply C by beta
            c *= beta;

            for k in 0..a.pattern().major_dim() {
                let a_row_k = a.get_lane(k).unwrap();
                for (&i, a_ki) in a_row_k.minor_indices().iter().zip(a_row_k.values()) {
                    let gamma_ki = alpha.clone() * a_ki.clone();
                    let mut c_row_i = c.row_mut(i);
                    match b {
                        Op::NoOp(ref b) => {
                            let b_row_k = b.row(k);
                            for (c_ij, b_kj) in c_row_i.iter_mut().zip(b_row_k.iter()) {
                                *c_ij += gamma_ki.clone() * b_kj.clone();
                            }
                        }
                        Op::Transpose(ref b) => {
                            let b_col_k = b.column(k);
                            for (c_ij, b_jk) in c_row_i.iter_mut().zip(b_col_k.iter()) {
                                *c_ij += gamma_ki.clone() * b_jk.clone();
                            }
                        }
                    }
                }
            }
        }
    }
}
