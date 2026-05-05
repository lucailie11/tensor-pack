use super::RawTensor;
use super::structure::broadcast_shape;

use std::ops::{Add, Sub, Mul, Div};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

use std::rc::Rc;

// Binary elementwise operations on two RawTensors with broadcasting support
// The core building block is elementwise_op
// Assign operations replace the left-hand side with the result (not in-place)
//
// Defined operations:
//   &RawTensor + &RawTensor  -> RawTensor       RawTensor += &RawTensor
//   &RawTensor - &RawTensor  -> RawTensor       RawTensor -= &RawTensor
//   &RawTensor * &RawTensor  -> RawTensor       RawTensor *= &RawTensor
//   &RawTensor / &RawTensor  -> RawTensor       RawTensor /= &RawTensor

impl RawTensor {
    // Panics if the shapes are incompatible for broadcasting
    pub fn elementwise_op(&self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) -> RawTensor {
        let out_shape = broadcast_shape(&self.shape, &other.shape);
        let a = self.expand(&out_shape);
        let b = other.expand(&out_shape);
        let new_data: Rc<[f64]> = a.iter().zip(b.iter())
            .map(|(x, y)| f(x, y))
            .collect(); 
        RawTensor::from_rc(&out_shape, new_data)
    }
}


impl Add for &RawTensor {
    type Output = RawTensor;

    fn add(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a + b)
    }
}

impl Sub for &RawTensor {
    type Output = RawTensor;

    fn sub(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a - b)
    }
}

impl Mul for &RawTensor {
    type Output = RawTensor;

    fn mul(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a * b)
    }
}
impl Div for &RawTensor {
    type Output = RawTensor;

    fn div(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a / b)
    }
}

impl AddAssign<&RawTensor> for RawTensor {
    fn add_assign(&mut self, other: &RawTensor) {
        *self = &*self + other;
    }
}

impl SubAssign<&RawTensor> for RawTensor {
    fn sub_assign(&mut self, other: &RawTensor) {
        *self = &*self - other;
    }
}

impl MulAssign<&RawTensor> for RawTensor {
    fn mul_assign(&mut self, other: &RawTensor) {
        *self = &*self * other;
    }
}

impl DivAssign<&RawTensor> for RawTensor {
    fn div_assign(&mut self, other: &RawTensor) {
        *self = &*self / other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_elementwise() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = RawTensor::from_slice(&[3], &[4.0, 5.0, 6.0]);
        let c = &a + &b;
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_elementwise() {
        let a = RawTensor::from_slice(&[3], &[4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let c = &a - &b;
        assert_eq!(c.data(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn mul_elementwise() {
        let a = RawTensor::from_slice(&[3], &[2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 2.0, 2.0]);
        let c = &a * &b;
        assert_eq!(c.data(), &[4.0, 6.0, 8.0]);
    }

    #[test]
    fn div_elementwise() {
        let a = RawTensor::from_slice(&[3], &[6.0, 8.0, 9.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 4.0, 3.0]);
        let c = &a / &b;
        assert_eq!(c.data(), &[3.0, 2.0, 3.0]);
    }

    #[test]
    fn add_elementwise_with_broadcast() {
        let a = RawTensor::from_slice(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[3, 2], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let c = &a + &b;
        assert_eq!(c.data(), &[5.0, 6.0, 8.0, 9.0, 11.0, 12.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0]);
    }

    #[test]
    #[should_panic]
    fn add_elementwise_with_broadcast_panic() {
        let a = RawTensor::from_slice(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let _c = &a + &b;
    }
}
