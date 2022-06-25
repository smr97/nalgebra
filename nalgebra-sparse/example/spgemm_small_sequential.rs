extern crate nalgebra_sparse;
use nalgebra_sparse::ops::filtered_sptgemm_sequential;
use nalgebra_sparse::CsrMatrix;
use std::fs::{self, DirEntry};
use std::io;
use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant};
const ROW_RANGE: Range<usize> = 0..1000;
//const COL_RANGE: Range<usize> = 0..100;

#[cfg(feature = "io")]
use nalgebra_sparse::io::load_coo_from_matrix_market_file;
fn main() {
    #[cfg(feature = "io")]
    {
        let mut file_iter = fs::read_dir("data").unwrap();
        for f in file_iter {
            println!("Benchmark file {:?}", f);
            let f = f.unwrap().path();

            if f.extension().map_or(false, |ext| ext == "mtx") {
                println!("Benchmark file {:?}", f);
                let sparse_input_matrix = load_coo_from_matrix_market_file::<f64, _>(&f).unwrap();
                let sparse_input_matrix = CsrMatrix::from(&sparse_input_matrix);
                let filtered_matrix = sparse_input_matrix.get_row_range(ROW_RANGE);
                let filtered_matrix_transpose = filtered_matrix.transpose();
                let now = Instant::now();
                let filtered_matrix = sparse_input_matrix.get_row_range(ROW_RANGE);
                let spmm_result_baseline = &filtered_matrix * &filtered_matrix_transpose;
                let spmm_time_baseline = now.elapsed().as_millis();
                let now = Instant::now();
                let spmm_result_spl_kernel = filtered_sptgemm_sequential(
                    &sparse_input_matrix.get_cs(),
                    &sparse_input_matrix.get_cs(),
                    ROW_RANGE,
                    ROW_RANGE,
                );
                let spmm_time = now.elapsed().as_millis();
                println!("handwritten SGEMM time was {}", spmm_time);
                println!("baseline SGEMM time was {}", spmm_time_baseline);
                let sum: f64 = spmm_result_baseline.triplet_iter().map(|(_, _, v)| v).sum();
                println!("sum of product baseline is {}", sum);
                let sum: f64 = spmm_result_spl_kernel.values().iter().sum();
                println!("sum of product handwritten is {}", sum);
            }
        }
    }
    #[cfg(not(feature = "io"))]
    {
        panic!("Run with IO feature only");
    }
}
