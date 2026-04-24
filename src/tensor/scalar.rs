use super::Tensor;
use crate::grad::BackpropOp;
use std::ops::{Add, Sub, Mul, Div};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

// Arithmetics between a RawTensor and a scalar f64
// Delegates data logic to RawTensor and autograd logic to grad/ 
// Assign operations replace the left-hand side with the result (not in-place)
//
// Defined operations:
//   &RawTensor + f64   -> RawTensor      f64 + &RawTensor  -> RawTensor      RawTensor += f64
//   &RawTensor - f64   -> RawTensor      f64 - &RawTensor  -> RawTensor      RawTensor -= f64
//   &RawTensor * f64   -> RawTensor      f64 * &RawTensor  -> RawTensor      RawTensor *= f64
//   &RawTensor / f64   -> RawTensor      f64 / &RawTensor  -> RawTensor      RawTensor /= f64


impl Add<&Tensor> for f64 {
    type Output = Tensor;

    fn add(self, tensor: &Tensor) -> Tensor {
        let raw = self + &tensor.raw;
        Tensor::autograd_tensor(raw, Box::from([tensor.clone()]), BackpropOp::AddScalar)
    }
}

impl Sub<&Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, tensor: &Tensor) -> Tensor {
        let raw = self - &tensor.raw;
        Tensor::autograd_tensor(raw, Box::from([tensor.clone()]), BackpropOp::SubScalar)
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Tensor {
        let raw = self * &tensor.raw;
        Tensor::autograd_tensor(raw, Box::from([tensor.clone()]), BackpropOp::MulScalar(self))
    }
}

impl Div<&Tensor> for f64 {
    type Output = Tensor;

    fn div(self, tensor: &Tensor) -> Tensor {
        let raw = self / &tensor.raw;
        Tensor::autograd_tensor(raw, Box::from([tensor.clone()]), BackpropOp::DivScalar(self))
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, scalar: f64) -> Tensor {
        scalar + self
    }
}


impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f64) -> Tensor {
        self + (-scalar)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Tensor {
        scalar * self
    }
}

impl Div<f64> for &Tensor {
    type Output = Tensor;

    fn div(self, scalar: f64) -> Tensor {
        self * (1.0 / scalar)
    }
}

impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, scalar: f64) {
        *self = &*self + scalar;
    }
}

impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, scalar: f64) {
        *self = &*self - scalar;
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, scalar: f64) {
        *self = &*self * scalar;
    }
}

impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, scalar: f64) {
        *self = &*self / scalar;
    }
}

