use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::fmt;

#[derive(Debug)]
pub struct Tensor {
    pub shape: Box<[usize]>,
    pub data: Box<[f64]>,
}

impl Tensor {
    // All constructors take the desired shape as a slice of dimension sizes.
    // Data is stored in row-major order: the last axis varies fastest.

    // Creates a Tensor field with 'value'
    pub fn full(shape: &[usize], value: f64) -> Tensor {
        let len: usize = shape.iter().product();
        Tensor {
            shape: Box::from(shape),
            data: vec![value; len].into_boxed_slice(),
        }
    }

    // Creates a Tensor filled with 0.0.
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    // Creates a Tensor filled with 1.0.
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    // Creates a 1D Tensor of `n` evenly spaced values from `start` to `end` (inclusive).
    pub fn linspace(start: f64, end: f64, n: usize) -> Tensor {
        assert!(n > 1, "n must be at least 2");

        let data: Vec<f64> = (0..n)
            .map(|i| start + i as f64 * (end - start) / (n - 1) as f64)
            .collect();

        Tensor {
            shape: vec![n].into_boxed_slice(),
            data: data.into_boxed_slice(),
        }
    }

    // Creates a Tensor filled with samples from N(`mean`, `std_dev`).
    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> Tensor {
        let len: usize = shape.iter().product();
        let mut rng = thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();
        let data: Vec<f64> = (0..len).map(|_| normal.sample(&mut rng)).collect();

        Tensor {
            shape: Box::from(shape),
            data: data.into_boxed_slice(),
        }
    }

    // Creates a Tensor by copying values from `data`. The product of `shape` must equal `data.len()`.
    pub fn new(shape: &[usize], data: &[f64]) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Data length doesn't match shape"
        );

        Tensor {
            shape: Box::from(shape),
            data: Box::from(data),
        }
    }

    // Changes the shape without moving data. The total number of elements must stay the same.
    pub fn reshape(&mut self, new_shape: &[usize]) {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.data.len(),
            "New shape doesn't match old data length"
        );
        self.shape = Box::from(new_shape);
    }
}

// Pretty-prints the tensor. 2D tensors are shown row-by-row; other ranks print the flat slice.
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Tensor {{")?;
        writeln!(f, "    shape: {:?}", self.shape)?;
        writeln!(f, "    data: ")?;
        if self.shape.len() == 2 {
            let cols = self.shape[1];
            for row in self.data.chunks(cols) {
                let formatted: Vec<String> = row.iter().map(|x| format!("{:.4}", x)).collect();
                writeln!(f, "           [{}]", formatted.join(", "))?;
            }
        } else {
            write!(f, "{:.4?}", self.data)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}
