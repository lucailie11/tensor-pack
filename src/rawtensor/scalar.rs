use super::RawTensor;

use std::ops::{Add, Sub, Mul, Div};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

// Arithmetics between a RawTensor and a scalar f64
// Every operation applies the scalar uniformly to all elements via map
// Assign operations replace the lhs with the result (not in-place)
//
// Defined operations:
//   &RawTensor + f64   -> RawTensor      f64 + &RawTensor  -> RawTensor      RawTensor += f64
//   &RawTensor - f64   -> RawTensor      f64 - &RawTensor  -> RawTensor      RawTensor -= f64
//   &RawTensor * f64   -> RawTensor      f64 * &RawTensor  -> RawTensor      RawTensor *= f64
//   &RawTensor / f64   -> RawTensor      f64 / &RawTensor  -> RawTensor      RawTensor /= f64

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
        *self = &*self + scalar;
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
        *self = &*self - scalar;
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
        *self = &*self * scalar;
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
        *self = &*self / scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let t = RawTensor::from_slice(&[4], &[1.0, 2.0, 4.0, 8.0]);
        assert_eq!(*(&t + 10.0).contiguous_data(), [11.0, 12.0, 14.0, 18.0]);
        assert_eq!(*(10.0 + &t).contiguous_data(), [11.0, 12.0, 14.0, 18.0]);
        assert_eq!(*(&t - 1.0).contiguous_data(), [0.0, 1.0, 3.0, 7.0]);
        assert_eq!(*(10.0 - &t).contiguous_data(), [9.0, 8.0, 6.0, 2.0]);
        assert_eq!(*(&t * 2.0).contiguous_data(), [2.0, 4.0, 8.0, 16.0]);
        assert_eq!(*(2.0 * &t).contiguous_data(), [2.0, 4.0, 8.0, 16.0]);
        assert_eq!(*(&t / 2.0).contiguous_data(), [0.5, 1.0, 2.0, 4.0]);
        assert_eq!(*(8.0 / &t).contiguous_data(), [8.0, 4.0, 2.0, 1.0]);
    }

    #[test]
    fn assign_ops() {
        let mut t = RawTensor::from_slice(&[4], &[1.0, 2.0, 4.0, 8.0]);
        t += 1.0;
        assert_eq!(*t.contiguous_data(), [2.0, 3.0, 5.0, 9.0]);
        t -= 1.0;
        assert_eq!(*t.contiguous_data(), [1.0, 2.0, 4.0, 8.0]);
        t *= 2.0;
        assert_eq!(*t.contiguous_data(), [2.0, 4.0, 8.0, 16.0]);
        t /= 2.0;
        assert_eq!(*t.contiguous_data(), [1.0, 2.0, 4.0, 8.0]);
    }
}

