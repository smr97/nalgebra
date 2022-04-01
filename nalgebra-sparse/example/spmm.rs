extern crate nalgebra_sparse;
use nalgebra_sparse::spmm_cs_prealloc_parallel;
use nalgebra_sparse::CsrMatrix;
use std::fs::{self, DirEntry};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(feature = "io")]
use nalgebra_sparse::io::load_coo_from_matrix_market_file;
fn main() {
    #[cfg(feature = "io")]
    {
        let mut file_iter = fs::read_dir("data/").unwrap();
        for f in file_iter {
            let f = f.unwrap().path();
            //println!("extension {:?}", f.extension());

            if f.extension().map_or(false, |ext| ext == "mtx") {
                println!("Benchmark file {:?}", f);
                let sparse_input_matrix = load_coo_from_matrix_market_file::<f64, _>(&f).unwrap();
                let sparse_input_matrix = CsrMatrix::from(&sparse_input_matrix);
                let spmm_result = &sparse_input_matrix * &sparse_input_matrix;
                let now = Instant::now();
                let spmm_result = &sparse_input_matrix * &sparse_input_matrix;
                let spmm_time = now.elapsed().as_millis();
                println!("SGEMM time was {}", spmm_time);
                let sum: f64 = spmm_result.triplet_iter().map(|(_, _, v)| v).sum();
                println!("sum of product is {}", sum);
                let now = Instant::now();
                let spmm_result_parallel = spmm_cs_prealloc_parallel::<f64>(
                    1.0,
                    &sparse_input_matrix.get_cs(),
                    &sparse_input_matrix.get_cs(),
                )
                .unwrap();
                let spmm_time = now.elapsed().as_millis();
                println!("SGEMM parallel time was {}", spmm_time);
                println!(
                    "Error {}%",
                    (spmm_result_parallel.values().into_iter().sum::<f64>()
                        - spmm_result.get_cs().values().into_iter().sum::<f64>())
                        / spmm_result.get_cs().values().into_iter().sum::<f64>()
                        * 100.0
                );
            }
        }
    }
    #[cfg(not(feature = "io"))]
    {
        panic!("Run with IO feature only");
    }
}
