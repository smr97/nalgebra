use std::collections::LinkedList;
//use std::time::Instant;

use crate::cs::CsMatrix;
use crate::ops::serial::{OperationError, OperationErrorKind};
use crate::ops::Op;
use crate::pattern::SparsityPatternFormatError;
use crate::SparseEntryMut;
use fxhash::FxHashMap;
use itertools::Itertools;
use nalgebra::{ClosedAdd, ClosedMul, DMatrixSlice, DMatrixSliceMut, Scalar};
use num_traits::{One, Zero};
//use rayon::prelude::*;
use rayon_logs::{prelude::*, subgraph, ThreadPoolBuilder};

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

/// First implementation of parallel SpGEMM
pub fn spmm_cs_prealloc_parallel<T>(
    alpha: T,
    a: &CsMatrix<T>,
    b: &CsMatrix<T>,
) -> Result<CsMatrix<T>, SparsityPatternFormatError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One + Send + Sync + Sized + Copy,
{
    let pool = ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("building pool failed");
    let num_rows = a.pattern().major_dim();
    //let now = Instant::now();
    //let (list_of_rows, runlog) =
    pool.compare()
        .runs_number(1)
        .attach_algorithm_nodisplay("parallel spgemm", || {
            (0..num_rows)
                .into_par_iter()
                .fold(
                    || (LinkedList::new()),
                    |mut list_of_hashes, a_row_id| {
                        let a_lane = a.get_lane(a_row_id).unwrap();
                        let _b_lane = b.get_lane(a_row_id);
                        let c_row_hash = a_lane
                        .indices_and_values()
                        //.minor_indices()
                        .into_par_iter()
                        //.zip(a_lane.values().into_par_iter())
                        .fold(
                            || FxHashMap::default(),
                            |mut scatter_values: FxHashMap<usize, T>, a_element_tuple| {
                                subgraph("b row iteration", 0, ||{
                                let (k, a_ik) = a_element_tuple;
                                let b_lane = b.get_lane(*k);
                                b_lane.map(|b_row| {
                                    b_row
                                        .minor_indices()
                                        .into_iter()
                                        .zip(b_row.values().into_iter())
                                        .for_each(|(j, b_kj)| {
                                            let some_entry =
                                                scatter_values.entry(*j).or_insert(Zero::zero());
                                            *some_entry +=
                                                alpha.clone() * a_ik.clone() * b_kj.clone();
                                        });
                                });
                                scatter_values
                            })
                        })
                        .reduce(
                            || FxHashMap::default(),
                            move |mut left_hash, right_hash| {
                                let new_right_hash =
                                    left_hash
                                        .drain()
                                        .fold(right_hash, |mut right_hash, (k, v)| {
                                            let right_entry =
                                                right_hash.entry(k).or_insert(Zero::zero());
                                            *right_entry += v;
                                            right_hash
                                        });
                                new_right_hash
                            },
                        );
                        //c_row_hash is now one full row of C
                        list_of_hashes.push_back(c_row_hash);
                        list_of_hashes
                    },
                )
                .map(|list_of_hashes| {
                    subgraph(
                        "hashmap to vecs",
                        list_of_hashes.iter().map(|map| map.len()).sum(),
                        || {
                            list_of_hashes.into_iter().fold(
                                LinkedList::new(),
                                |mut list: LinkedList<(Vec<usize>, Vec<T>)>, mut c_row_hash| {
                                    list.push_back(
                                        c_row_hash
                                            .drain()
                                            .sorted_unstable_by(|left, right| left.0.cmp(&right.0))
                                            .unzip(),
                                    );
                                    list
                                },
                            )
                        },
                    )
                })
                .reduce(
                    || LinkedList::new(),
                    |mut left_list, mut right_list| {
                        left_list.append(&mut right_list);
                        left_list
                    },
                );
        })
        .generate_logs("report.html")
        .expect("error saving report");
    //runlog
    //    .save_svg("./parallel_run.svg")
    //    .expect("couldn't save log");
    panic!("not implemented");
    //let parallel_time = now.elapsed();
    //let now = Instant::now();
    //let offsets = list_of_rows.iter().fold(vec![0], |mut offsets, row| {
    //    offsets.push(offsets.last().map_or(row.0.len(), |offset_correction| {
    //        offset_correction + row.0.len()
    //    }));
    //    offsets
    //});
    //let indices: Vec<_> = list_of_rows
    //    .iter()
    //    .map(|(indices, _values)| indices)
    //    .flatten()
    //    .cloned()
    //    .collect();

    //let values: Vec<_> = list_of_rows
    //    .iter()
    //    .map(|(_indices, values)| values)
    //    .flatten()
    //    .cloned()
    //    .collect();
    //let collections_time = now.elapsed();
    //println!(
    //    "Parallel time was {:?}, collections time was {:?}",
    //    parallel_time, collections_time
    //);

    //SparsityPattern::try_from_offsets_and_indices(
    //    num_rows,
    //    b.pattern().minor_dim(),
    //    offsets,
    //    indices,
    //)
    //.map(|pattern| CsMatrix::from_pattern_and_values(pattern, values))
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
