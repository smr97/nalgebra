use itertools::{EitherOrBoth::Both, Itertools};
//use itertools::Itertools;
use nalgebra::{ClosedAdd, ClosedMul, Scalar};
use num_traits::{One, Zero};
use std::{cmp::max, ops::Range, time::Instant};

//use crate::cs::CsMatrix;
//use crate::ops::serial::OperationError;
//use crate::ops::Op;
//use crate::SparseEntryMut;
use kvik::prelude::{Divisible, True};

use crate::{cs::CsMatrix, pattern::SparsityPattern};
//use nalgebra::{ClosedAdd, ClosedMul, Scalar};
//use num_traits::{One, Zero};

const NUM_DWORDS_IN_L2: usize = 524288;

/// Task struct for the product:- result = left * right.
/// Represents the task in the form of two pointers for each input matrix, start/end for each
/// matrix.
/// As we cut the pointers in half, we are basically cutting the dense space in half each time.
///
/// For a quick cut on the right side, it ASSUMES that right_matrix is transposed.
struct SpGEMMTask<'a, T> {
    left_matrix: &'a CsMatrix<T>,
    left_start: usize, // TODO change this to range, noob.
    left_end: usize,
    right_matrix: &'a CsMatrix<T>,
    right_start: usize,
    right_end: usize,
    //result_matrix: CsMatrix<T>,
}

impl<'a, T> Divisible for SpGEMMTask<'a, T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One + Clone,
{
    type Controlled = True;
    // Don't divide iff result can be fit in cache
    fn should_be_divided(&self) -> bool {
        let num_cols = self.right_end - self.right_start + 1; // oh so precise
        let num_rows = self.left_end - self.left_start + 1;
        let estimated_size = max(num_rows, num_cols); //Assume nnz is order of M or N.
        estimated_size >= NUM_DWORDS_IN_L2
    }
    fn divide(self) -> (Self, Self) {
        let num_cols = self.right_end - self.right_start + 1; // oh so precise
        let num_rows = self.left_end - self.left_start + 1;
        // skinny or fat?
        if num_rows > num_cols {
            //cut only the left matrix on rows
            let top_task = SpGEMMTask {
                left_matrix: self.left_matrix,
                right_matrix: self.right_matrix,
                //result_matrix: self.result_matrix,
                left_start: self.left_start,
                left_end: (self.left_start + self.left_end) / 2,
                right_start: self.right_start,
                right_end: self.right_end,
            };
            let bottom_task = SpGEMMTask {
                left_matrix: self.left_matrix,
                right_matrix: self.right_matrix,
                //result_matrix: CsMatrix::new(
                //    self.left_end - (self.left_end + self.left_start) / 2 + 1,
                //    self.right_end - self.right_start + 1,
                //),
                left_end: self.left_end,
                left_start: (self.left_start + self.left_end) / 2 + 1,
                right_start: self.right_start,
                right_end: self.right_end,
            };

            (top_task, bottom_task)
        } else {
            //cut right matrix on cols only
            //TODO check for crossover errors. the bounds may be wrong.
            let left_task = SpGEMMTask {
                left_matrix: self.left_matrix,
                right_matrix: self.right_matrix,
                //result_matrix: self.result_matrix,
                left_start: self.left_start,
                left_end: self.left_end,
                right_start: self.right_start,
                right_end: (self.right_start + self.right_end) / 2,
            };
            let right_task = SpGEMMTask {
                left_matrix: self.left_matrix,
                right_matrix: self.right_matrix,
                //result_matrix: CsMatrix::new(
                //    self.left_end - self.left_start + 1,
                //    self.right_end - (self.right_end + self.right_start) / 2 + 1,
                //),
                left_start: self.left_start,
                left_end: self.left_end,
                right_end: self.right_end,
                right_start: (self.right_start + self.right_end) / 2 + 1,
            };
            (left_task, right_task)
        }
    }
    fn divide_at(self, _index: usize) -> (Self, Self) {
        unimplemented!("Not sure why we need to divide SpGEMM at a given point, but we can write this based on output size pretty easily")
    }
}

impl<'a, T> Iterator for SpGEMMTask<'a, T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    type Item = CsMatrix<T>; // We probably also need to maintain some pointers to know where this fits in the big matrix
                             // This call should finish the given SpGEMMTask sequentially
                             // We need a fast SpGEMM kernel specialised for one-shot multiplies.
                             // Do away with symbolic + actual computation.
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_start == self.left_end || self.right_start == self.right_end {
            None
        } else {
            Some(filtered_sptgemm_sequential(
                self.left_matrix,
                self.right_matrix,
                self.left_start..self.left_end,
                self.right_start..self.right_end,
            ))
        }
    }
}

///multiply some rows of left matrix with some columns of right matrix, as given by the two ranges.
///assume that right matrix is transposed.
/// OUTPUT: Allocates memory and returns a small sparse matrix that represents this task output.
///         The output matrix indices will be GLOBAL indices however. Ie, they will lie within
///         right_range.
pub fn filtered_sptgemm_sequential<'a, T>(
    left: &'a CsMatrix<T>,
    right: &'a CsMatrix<T>,
    left_range: Range<usize>,
    right_range: Range<usize>,
) -> CsMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    let mut offsets = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    let left_range_copy = left_range.clone();
    let right_range_copy = right_range.clone();
    let mut compute_time = 0u128;

    for row_index in left_range {
        let right_range_copy = right_range.clone();
        for col_index in right_range_copy {
            let left_lane = left.get_lane(row_index).unwrap();
            let right_lane = right.get_lane(col_index).unwrap();
            let left_lane_iter = left_lane
                .minor_indices()
                .iter()
                .zip(left_lane.values().iter());
            let right_lane_iter = right_lane
                .minor_indices()
                .iter()
                .zip(right_lane.values().iter());
            let start = Instant::now();
            let output: T = left_lane_iter
                .merge_join_by(right_lane_iter, |(left_pos, _), (right_pos, _)| {
                    left_pos.cmp(right_pos)
                })
                .map(|some_element| {
                    if let Both(left_tuple, right_tuple) = some_element {
                        left_tuple.1.clone() * right_tuple.1.clone()
                    } else {
                        Zero::zero()
                    }
                })
                .fold(Zero::zero(), |curr_sum, val| curr_sum + val);
            compute_time += start.elapsed().as_nanos();

            if !output.is_zero() {
                values.push(output);
                indices.push(col_index); //The index array in the output matrix refers to global column indices then.
            }
        }
        offsets.push(indices.len());
    }
    println!("Compute time was {:?}", compute_time / 1_000_000u128);

    unsafe {
        let output_sparsity_pattern = SparsityPattern::from_offset_and_indices_unchecked(
            left_range_copy.end - left_range_copy.start,
            right_range_copy.end - right_range_copy.start,
            offsets,
            indices,
        );

        CsMatrix::from_pattern_and_values(output_sparsity_pattern, values)
    }
}

//impl<'a, T> Producer for SpGEMMTask<'a, T> where T: Clone + Sync + Send {}

//pub fn spmm_cs_parallel<T>(
//    beta: T,
//    c: &mut CsMatrix<T>,
//    alpha: T,
//    a: &CsMatrix<T>,
//    b: &CsMatrix<T>,
//) -> Result<(), OperationError>
//where
//    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
//{
//    unimplemented!("Not yet built");
//    //assert_eq!(c.pattern().major_dim(), a.pattern().major_dim());
//    //assert_eq!(c.pattern().minor_dim(), b.pattern().minor_dim());
//    //let some_val = Zero::zero();
//    //let mut scratchpad_values: Vec<T> = vec![some_val; b.pattern().minor_dim()];
//    //for i in 0..c.pattern().major_dim() {
//    //    let a_lane_i = a.get_lane(i).unwrap();
//
//    //    let mut c_lane_i = c.get_lane_mut(i).unwrap();
//
//    //    for (&k, a_ik) in a_lane_i.minor_indices().iter().zip(a_lane_i.values()) {
//    //        let b_lane_k = b.get_lane(k).unwrap();
//    //        let alpha_aik = alpha.clone() * a_ik.clone();
//    //        for (j, b_kj) in b_lane_k.minor_indices().iter().zip(b_lane_k.values()) {
//    //            // use a dense scatter vector to accumulate non-zeros quickly
//    //            unsafe {
//    //                *scratchpad_values.get_unchecked_mut(*j) += alpha_aik.clone() * b_kj.clone();
//    //            }
//    //        }
//    //    }
//
//    //    //Get indices from C pattern and gather from the dense scratchpad_values
//    //    let (indices, values) = c_lane_i.indices_and_values_mut();
//    //    values
//    //        .iter_mut()
//    //        .zip(indices)
//    //        .for_each(|(output_ref, index)| unsafe {
//    //            *output_ref = beta.clone() * output_ref.clone()
//    //                + scratchpad_values.get_unchecked(*index).clone();
//    //            *scratchpad_values.get_unchecked_mut(*index) = Zero::zero();
//    //        });
//    //}
//
//    //Ok(())
//}
