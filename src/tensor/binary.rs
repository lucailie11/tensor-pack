use super::Tensor;
use crate::grad::BackpropOp;
use std::ops::{Add, Sub, Mul, Div};

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        let raw = &self.raw + &other.raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::AddTensor)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        let raw = &self.raw - &other.raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::SubTensor)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        let raw = &self.raw * &other.raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::MulTensor)
    }
}


impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        let raw = &self.raw / &other.raw;
        Tensor::tensor_grad(raw, Box::from([self.clone(), other.clone()]), BackpropOp::DivTensor)
    }
}

// impl AddAssign<&Tensor> for Tensor {
//     fn add_assign(&mut self, other: &Tensor) {
//         self.0.raw += &other.raw;
//     }
// }
//
// impl SubAssign<&Tensor> for Tensor {
//     fn sub_assign(&mut self, other: &Tensor) {
//         self.elementwise_op_inplace(other, |a, b| a - b);
//     }
// }
// impl MulAssign<&Tensor> for Tensor {
//     fn mul_assign(&mut self, other: &Tensor) {
//         self.elementwise_op_inplace(other, |a, b| a * b);
//     }
// }

// impl DivAssign<&Tensor> for Tensor {
//     fn div_assign(&mut self, other: &Tensor) {
//         self.elementwise_op_inplace(other, |a, b| a / b);
//     }
// }
//
