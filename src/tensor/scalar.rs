use super::Tensor;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, scalar: f64) -> Tensor {
        self.map(|x| x + scalar)
    }
}

impl Add<Tensor> for f64 {
    type Output = Tensor;

    fn add(self, tensor: Tensor) -> Tensor {
        tensor.map(|x| x + self)
    }
}

impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x + scalar)
    }
}

impl Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f64) -> Tensor {
        self.map(|x| x - scalar)
    }
}

impl Sub<Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, tensor: Tensor) -> Tensor {
        tensor.map(|x| self - x)
    }
}

impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x - scalar);
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Tensor {
        self.map(|x| x * scalar)
    }
}

impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, tensor: Tensor) -> Tensor {
        tensor.map(|x| x * self)
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x * scalar);
    }
}

impl Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, scalar: f64) -> Tensor {
        self.map(|x| x / scalar)
    }
}

impl Div<Tensor> for f64 {
    type Output = Tensor;

    fn div(self, tensor: Tensor) -> Tensor {
        tensor.map(|x| self / x)
    }
}

impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x / scalar);
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn add_scalar() {
        let a = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = a + 1.0;
        assert_eq!(&*b.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn scalar_add_tensor() {
        let a = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = 1.0 + a;
        assert_eq!(&*b.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn mul_scalar() {
        let a = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = a * 2.0;
        assert_eq!(&*b.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn div_scalar() {
        let a = Tensor::from_slice(&[3], &[2.0, 4.0, 6.0]);
        let b = a / 2.0;
        assert_eq!(&*b.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sub_scalar() {
        let a = Tensor::from_slice(&[3], &[3.0, 4.0, 5.0]);
        let b = a - 1.0;
        assert_eq!(&*b.data(), &[2.0, 3.0, 4.0]);
    }
}
