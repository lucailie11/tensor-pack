use crate::Tensor;
use std::ops::{Add, AddAssign};

/*
 * Addition between two &Tensor requires them to have the same total length
 * (not the same shape) and adds them pointwise
*/

/*
 * Addition between a &Tensor and a f64 adds the scalar to
 * all elements of the Tensor
*/

/*
 * Defined operations
 * &Tensor + &Tensor -> Tensor
 * Tensor += &Tensor
 * &Tensor + f64
 * &Tensor += f64
 * f64 + &Tensor
*/

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

        Tensor {
            shape: self.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
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

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, other: f64) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a + other).collect();
        Tensor {
            shape: self.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
    }
}

impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, other: f64) {
        for a in self.data.iter_mut() {
            *a += other;
        }
    }
}

impl Add<&Tensor> for f64 {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        other + self
    }
}
