use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::fmt;

#[derive(Debug)]
pub struct Tensor {
    pub shape: Box<[usize]>,
    pub data: Box<[f64]>,
}

impl Tensor {
    // All functions take the shape of the new Tensor unless specified otherwise

    // Creates a Tensor filled with the same value
    pub fn full(shape: &[usize], value: f64) -> Tensor {
        let len: usize = shape.iter().product();
        Tensor {
            shape: Box::from(shape),
            data: vec![value; len].into_boxed_slice(),
        }
    }

    // Creates a Tensor filled with 0
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    // Creates a Tensor filled with 1
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    // Creates an 1D Tensor with n values in progression from start to end
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

    // Creates a Tensor with values from a slice (by copying them)
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

    // Creates a Tensor with random values from a normal distribution
    pub fn randn(shape: &[usize], mean: f64, variance: f64) -> Tensor {
        let len: usize = shape.iter().product();
        let mut rng = thread_rng();
        let normal = Normal::new(mean, variance.sqrt()).unwrap();
        let data: Vec<f64> = (0..len).map(|_| normal.sample(&mut rng)).collect();

        Tensor {
            shape: Box::from(shape),
            data: data.into_boxed_slice(),
        }
    }

    // Reshapes a Tensor
    pub fn reshape(&mut self, new_shape: &[usize]) {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.data.len(),
            "New shape doesn't match old data length"
        );
        self.shape = Box::from(new_shape);
    }
}

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
