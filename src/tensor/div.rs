use crate::Tensor;
use std::ops::{Div, DivAssign};

impl Div<f64> for &Tensor {
    type Output = Tensor;

    fn div(self, other: f64) -> Tensor {
        self * (1.0 / other)
    }
}

impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, other: f64) {
        *self *= 1.0 / other;
    }
}
