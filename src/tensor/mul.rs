use crate::Tensor;
use std::ops::Mul;

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
