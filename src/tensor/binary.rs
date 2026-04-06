use super::Tensor;
use crate::grad::BackpropOp;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        let raw = &self.borrow().raw + &other.borrow().raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::AddTensor)
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        self.borrow_mut().raw += &other.borrow().raw;
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        let raw = &self.borrow().raw - &other.borrow().raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::SubTensor)
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        self.borrow_mut().raw -= &other.borrow().raw;
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        let raw = &self.borrow().raw * &other.borrow().raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::MulTensor)
    }
}

impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        self.borrow_mut().raw *= &other.borrow().raw;
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        let raw = &self.borrow().raw / &other.borrow().raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::DivTensor)
    }
}

impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        self.borrow_mut().raw += &other.borrow().raw;
    }
}
