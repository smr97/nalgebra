use crate::cs::CsMatrix;
use crate::ops::serial::{OperationError, OperationErrorKind};
use crate::ops::Op;
use crate::pattern::{SparsityPattern, SparsityPatternFormatError};
use crate::SparseEntryMut;
use fxhash::FxHashMap;
use itertools::{EitherOrBoth, Itertools};
use nalgebra::{ClosedAdd, ClosedMul, DMatrixSlice, DMatrixSliceMut, Scalar};
use num_traits::{One, Zero};
use rayon::prelude::*;

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
    let mut scratchpad_indices: Vec<usize> = vec![0; b.pattern().minor_dim()];
    let mut scratchpad_used: Vec<bool> = vec![false; b.pattern().minor_dim()];
    let mut right_end = 0usize;
    for i in 0..c.pattern().major_dim() {
        let a_lane_i = a.get_lane(i).unwrap();

        let mut c_lane_i = c.get_lane_mut(i).unwrap();

        for (&k, a_ik) in a_lane_i.minor_indices().iter().zip(a_lane_i.values()) {
            let b_lane_k = b.get_lane(k).unwrap();
            let alpha_aik = alpha.clone() * a_ik.clone();
            for (j, b_kj) in b_lane_k.minor_indices().iter().zip(b_lane_k.values()) {
                // Determine the location in C to append the value
                scratchpad_values[*j] += alpha_aik.clone() * b_kj.clone();
                if !scratchpad_used[*j] {
                    scratchpad_indices[right_end] = *j;
                    right_end += 1;
                    scratchpad_used[*j] = true;
                }
            }
        }
        // sort the indices, and then access the relevant indices (in sorted order) from values
        // into C.
        scratchpad_indices[0..right_end].sort_unstable();
        c_lane_i
            .values_mut()
            .iter_mut()
            .zip(scratchpad_indices[0..right_end].iter())
            .for_each(|(output_ref, index)| {
                *output_ref = beta.clone() * output_ref.clone() + scratchpad_values[*index].clone();
                scratchpad_used[*index] = false;
                scratchpad_values[*index] = Zero::zero();
            });
        right_end = 0usize;
    }

    Ok(())
}

/// First implementation of parallel SpGEMM
pub fn spmm_cs_prealloc_parallel<T>(
    //c: &mut CsMatrix<T>,
    alpha: T,
    a: &CsMatrix<T>,
    b: &CsMatrix<T>,
) -> Result<CsMatrix<T>, SparsityPatternFormatError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One + Send + Sync + Sized + Copy,
{
    let num_rows = a.pattern().major_dim();
    let (mut offset, indices, values) = (0..num_rows)
        .into_par_iter()
        .map(|row_id| {
            //TODO lanes may not exist? Handle the unwrap panics.
            let a_lane = a.get_lane(row_id).unwrap();
            let _b_lane = b.get_lane(row_id);
            let (indices, values) = a_lane
                .minor_indices()
                .into_par_iter()
                .zip(a_lane.values().into_par_iter())
                .map(|(k, a_ik)| {
                    let mut scatter_values = FxHashMap::default();
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
                    (
                        scatter_values.keys().sorted().cloned().collect_vec(),
                        scatter_values,
                    )
                })
                .reduce_with(move |left_tuple, right_tuple| {
                    let (left_indices, mut left_values) = left_tuple;
                    let (right_indices, right_values) = right_tuple;
                    let total_indices = left_indices
                        .iter()
                        .merge_join_by(right_indices.iter(), |l, r| l.cmp(r))
                        .map(|some_val| match some_val {
                            EitherOrBoth::Right(r) => {
                                left_values.insert(*r, right_values[r]);
                                r.clone()
                            }
                            EitherOrBoth::Both(l, r) => {
                                left_values.get_mut(l).map(|val| *val += right_values[r]);
                                l.clone()
                            }
                            EitherOrBoth::Left(l) => l.clone(),
                        })
                        .collect_vec();

                    (total_indices, left_values)
                })
                .unwrap();
            // Now we have computed one row of C in parallel.
            // Return a sparse vector with offset, indices, values
            let non_zeros = indices.iter().map(move |index| values[index]).collect_vec();
            (vec![indices.len()], indices, non_zeros)
        })
        .reduce_with(|left_triplet, right_triplet| {
            let (mut left_offset, mut left_indices, mut left_values) = left_triplet;
            let (mut right_offset, right_indices, right_values) = right_triplet;
            //offsets need some kind of a prefix sum
            let offset_correction = left_offset.last().unwrap();
            right_offset
                .iter_mut()
                .for_each(|v| *v += offset_correction);
            left_offset.extend(right_offset);
            //indices mostly just need to be extended
            left_indices.extend(right_indices);
            //values were compressed in the previous reduction to have only the non-zeros.
            //concatenate them just like that
            left_values.extend(right_values);
            (left_offset, left_indices, left_values)
        })
        .unwrap();
    offset.insert(0, 0);
    SparsityPattern::try_from_offsets_and_indices(
        num_rows,
        b.pattern().minor_dim(),
        offset,
        indices,
    )
    .map(|pattern| CsMatrix::from_pattern_and_values(pattern, values))
    //Ok(())
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
