use super::RawTensor;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

// Arithmetic between a RawTensor and a scalar f64.
// Every operation applies the scalar uniformly to all elements via map / map_inplace.
//
// Defined operations:
//   &RawTensor + f64   -> RawTensor      f64 + &RawTensor  -> RawTensor
//    RawTensor += f64
//   &RawTensor - f64   -> RawTensor      f64 - &RawTensor  -> RawTensor
//    RawTensor -= f64
//   &RawTensor * f64   -> RawTensor      f64 * &RawTensor  -> RawTensor
//    RawTensor *= f64
//   &RawTensor / f64   -> RawTensor      f64 / &RawTensor  -> RawTensor
//    RawTensor /= f64

impl Add<f64> for &RawTensor {
    type Output = RawTensor;

    fn add(self, scalar: f64) -> RawTensor {
        self.map(|x| x + scalar)
    }
}

impl Add<&RawTensor> for f64 {
    type Output = RawTensor;

    fn add(self, tensor: &RawTensor) -> RawTensor {
        tensor.map(|x| x + self)
    }
}

impl AddAssign<f64> for RawTensor {
    fn add_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x + scalar)
    }
}

impl Sub<f64> for &RawTensor {
    type Output = RawTensor;

    fn sub(self, scalar: f64) -> RawTensor {
        self.map(|x| x - scalar)
    }
}

impl Sub<&RawTensor> for f64 {
    type Output = RawTensor;

    fn sub(self, tensor: &RawTensor) -> RawTensor {
        tensor.map(|x| self - x)
    }
}

impl SubAssign<f64> for RawTensor {
    fn sub_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x - scalar);
    }
}

impl Mul<f64> for &RawTensor {
    type Output = RawTensor;

    fn mul(self, scalar: f64) -> RawTensor {
        self.map(|x| x * scalar)
    }
}

impl Mul<&RawTensor> for f64 {
    type Output = RawTensor;

    fn mul(self, tensor: &RawTensor) -> RawTensor {
        tensor.map(|x| x * self)
    }
}

impl MulAssign<f64> for RawTensor {
    fn mul_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x * scalar);
    }
}

impl Div<f64> for &RawTensor {
    type Output = RawTensor;

    fn div(self, scalar: f64) -> RawTensor {
        self.map(|x| x / scalar)
    }
}

impl Div<&RawTensor> for f64 {
    type Output = RawTensor;

    fn div(self, tensor: &RawTensor) -> RawTensor {
        tensor.map(|x| self / x)
    }
}

impl DivAssign<f64> for RawTensor {
    fn div_assign(&mut self, scalar: f64) {
        self.map_inplace(|x| x / scalar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_scalar() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = &a + 1.0;
        assert_eq!(b.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn scalar_add_tensor() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = 1.0 + &a;
        assert_eq!(b.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn sub_scalar() {
        let a = RawTensor::from_slice(&[3], &[3.0, 4.0, 5.0]);
        let b = &a - 1.0;
        assert_eq!(b.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn mul_scalar() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = &a * 2.0;
        assert_eq!(b.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn div_scalar() {
        let a = RawTensor::from_slice(&[3], &[2.0, 4.0, 6.0]);
        let b = &a / 2.0;
        assert_eq!(b.data(), &[1.0, 2.0, 3.0]);
    }
}
