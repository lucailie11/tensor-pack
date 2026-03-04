use crate::Tensor;
use std::ops::{Sub, SubAssign};

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch");

        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Tensor {
            shape: self.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch");

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b;
        }
    }
}

impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: f64) -> Tensor {
        self + (-other)
    }
}

impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, other: f64) {
        *self += -other;
    }
}

impl Sub<&Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        let new_data: Vec<f64> = other.data.iter().map(|a| self - a).collect();

        Tensor {
            shape: other.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
    }
}
