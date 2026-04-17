use super::Tensor;
use crate::grad::BackpropOp;
use std::ops::{Add, Sub, Mul, Div};

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


// impl AddAssign<f64> for Tensor {
//     fn add_assign(&mut self, scalar: f64) {
//         self.map_inplace(|x| x + scalar)
//     }
// }
//
// impl SubAssign<f64> for Tensor {
//     fn sub_assign(&mut self, scalar: f64) {
//         self.map_inplace(|x| x - scalar);
//     }
// }
//
// impl MulAssign<f64> for Tensor {
//     fn mul_assign(&mut self, scalar: f64) {
//         self.map_inplace(|x| x * scalar);
//     }
// }
//
// impl DivAssign<f64> for Tensor {
//     fn div_assign(&mut self, scalar: f64) {
//         self.map_inplace(|x| x / scalar);
//     }
// }
//
