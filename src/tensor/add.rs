use crate::Tensor;
use std::ops::{Add, AddAssign};

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch");

        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor::new(&self.shape, &new_data)
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, other: f64) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a + other).collect();
        Tensor::new(&self.shape, &new_data)
    }
}

impl Add<&Tensor> for f64 {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        other + self
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch");

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }
}
