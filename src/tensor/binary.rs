use super::Tensor;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

impl Tensor {
    pub fn elementwise_op(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Tensor {
        let raw = self.data.borrow().elementwise_op(&other.data.borrow(), f);
        Tensor::new_tensor(raw, None)
    }

    pub fn elementwise_op_inplace(&mut self, other: &Tensor, f: impl Fn(f64, f64) -> f64) {
        self.data.borrow_mut().elementwise_op_inplace(&other.data.borrow(), f);
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        self.elementwise_op(&other, |a, b| a + b)
    }
}

impl AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        self.elementwise_op_inplace(&other, |a, b| a + b);
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        self.elementwise_op(&other, |a, b| a - b)
    }
}

impl SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        self.elementwise_op_inplace(&other, |a, b| a - b);
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        self.elementwise_op(&other, |a, b| a * b)
    }
}

impl MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, other: Tensor) {
        self.elementwise_op_inplace(&other, |a, b| a * b);
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        self.elementwise_op(&other, |a, b| a / b)
    }
}

impl DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, other: Tensor) {
        self.elementwise_op_inplace(&other, |a, b| a / b);
    }
}
