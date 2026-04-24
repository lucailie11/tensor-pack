use super::Tensor;
use crate::grad::BackpropOp;
use std::ops::{Add, Sub, Mul, Div};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

// Binary elementwise operations on two Tensors with broadcasting support
// Delegates data logic to RawTensor and autograd logic to grad/ 
// Assign operations replace the left-hand side with the result (not in-place)
//
// Defined operations:
//   &RawTensor + &RawTensor  -> RawTensor       RawTensor += &RawTensor
//   &RawTensor - &RawTensor  -> RawTensor       RawTensor -= &RawTensor
//   &RawTensor * &RawTensor  -> RawTensor       RawTensor *= &RawTensor
//   &RawTensor / &RawTensor  -> RawTensor       RawTensor /= &RawTensor

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        let raw = &self.raw + &other.raw;
        Tensor::autograd_tensor(raw, Box::from([self.clone(), other.clone()]), BackpropOp::AddTensor)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        let raw = &self.raw - &other.raw;
        Tensor::autograd_tensor(raw, Box::from([self.clone(), other.clone()]), BackpropOp::SubTensor)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        let raw = &self.raw * &other.raw;
        Tensor::autograd_tensor(raw, Box::from([self.clone(), other.clone()]), BackpropOp::MulTensor)
    }
}


impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        let raw = &self.raw / &other.raw;
        Tensor::autograd_tensor(raw, Box::from([self.clone(), other.clone()]), BackpropOp::DivTensor)
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        *self = &*self + other;
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        *self = &*self - other;
    }
}
impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        *self = &*self * other;
    }
}

impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        *self = &*self / other;
    }
}

