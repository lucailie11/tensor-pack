use crate::Tensor;
use std::ops::{Mul, MulAssign};

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: f64) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a * other).collect();
        Tensor::new(&self.shape, &new_data)
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        other * self
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, other: f64) {
        for a in self.data.iter_mut() {
            *a *= other;
        }
    }
}
