use crate::Tensor;
use std::ops::{Mul, MulAssign};

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|a| a * other).collect();
        Tensor {
            shape: self.shape.clone(),
            data: data.into_boxed_slice(),
        }
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, other: f64) {
        for a in self.data.iter_mut() {
            *a *= other;
        }
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        other * self
    }
}
