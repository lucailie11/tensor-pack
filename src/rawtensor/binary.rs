use super::RawTensor;
use super::structure::get_broadcast_shape;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};
use std::rc::Rc;

// Binary elementwise operations on two Tensors with broadcasting support
// Inplace variants require self to already hold the output shape
//
// The core building blocks are elementwise_op and elementwise_op_inplace.
//
// Defined operations:
//   &RawTensor + &RawTensor  -> RawTensor
//    RawTensor += &RawTensor
//   &RawTensor - &RawTensor  -> RawTensor
//    RawTensor -= &RawTensor
//   &RawTensor * &RawTensor  -> RawTensor
//    RawTensor *= &RawTensor
//   &RawTensor / &RawTensor  -> RawTensor
//    RawTensor /= &RawTensor

impl RawTensor {
    // Panics if the shapes are incompatible for broadcasting
    pub fn elementwise_op(&self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) -> RawTensor {
        let out_shape = get_broadcast_shape(&self.shape, &other.shape);
        let a = self.expand(&out_shape);
        let b = other.expand(&out_shape);
        let new_data: Box<[f64]> = a.iter_strided()
            .zip(b.iter_strided())
            .map(|(x, y)| f(x, y))
            .collect();
        RawTensor::from_box(&out_shape, new_data)
    }

    // Panics if self doesn't already have the broadcast output shape
    pub fn elementwise_op_inplace(&mut self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) {
        assert_eq!(
            get_broadcast_shape(&self.shape, &other.shape),
            self.shape,
            "in-place broadcasting requires self to already be the output shape"
        );

        let b = other.expand(&self.shape);
        let data = Rc::get_mut(&mut self.data)
            .expect("cannot modify tensor in-place: multiple owners");
        data.iter_mut().zip(b.iter_strided()).for_each(|(x, y)| *x = f(*x, y));
    }
}


impl Add for &RawTensor {
    type Output = RawTensor;

    fn add(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a + b)
    }
}

impl AddAssign<&RawTensor> for RawTensor {
    fn add_assign(&mut self, other: &RawTensor) {
        self.elementwise_op_inplace(other, |a, b| a + b);
    }
}

impl Sub for &RawTensor {
    type Output = RawTensor;

    fn sub(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a - b)
    }
}

impl SubAssign<&RawTensor> for RawTensor {
    fn sub_assign(&mut self, other: &RawTensor) {
        self.elementwise_op_inplace(other, |a, b| a - b);
    }
}

impl Mul for &RawTensor {
    type Output = RawTensor;

    fn mul(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a * b)
    }
}

impl MulAssign<&RawTensor> for RawTensor {
    fn mul_assign(&mut self, other: &RawTensor) {
        self.elementwise_op_inplace(other, |a, b| a * b);
    }
}

impl Div for &RawTensor {
    type Output = RawTensor;

    fn div(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a / b)
    }
}

impl DivAssign<&RawTensor> for RawTensor {
    fn div_assign(&mut self, other: &RawTensor) {
        self.elementwise_op_inplace(other, |a, b| a / b);
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
    fn add_assign() {
        let mut a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = RawTensor::from_slice(&[3], &[1.0, 1.0, 1.0]);
        a += &b;
        assert_eq!(a.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn sub_assign() {
        let mut a = RawTensor::from_slice(&[3], &[4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        a -= &b;
        assert_eq!(a.data(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn mul_assing() {
        let mut a = RawTensor::from_slice(&[3], &[2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 2.0, 2.0]);
        a *= &b;
        assert_eq!(a.data(), &[4.0, 6.0, 8.0]);
    }

    #[test]
    fn div_assign() {
        let mut a = RawTensor::from_slice(&[3], &[6.0, 8.0, 9.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 4.0, 3.0]);
        a /= &b;
        assert_eq!(a.data(), &[3.0, 2.0, 3.0]);
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

    #[test]
    fn add_assign_with_broadcast() {
        let mut a = RawTensor::from_slice(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[3, 1], &[4.0, 5.0, 6.0]);
        a += &b;
        assert_eq!(a.shape(), &[2, 3, 1]);
        assert_eq!(a.data(), &[5.0, 7.0, 9.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic]
    fn add_assign_with_broadcast_panic() {
        let mut a = RawTensor::from_slice(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[3, 2], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        a += &b;
    }

}
