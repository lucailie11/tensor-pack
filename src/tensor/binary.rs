use super::Tensor;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

impl Tensor {
    pub fn elementwise_op(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Tensor {
        let raw = self.raw.borrow().elementwise_op(&other.raw.borrow(), f);
        Tensor::from_raw(raw, None)
    }

    pub fn elementwise_op_inplace(&mut self, other: &Tensor, f: impl Fn(f64, f64) -> f64) {
        self.raw.borrow_mut().elementwise_op_inplace(&other.raw.borrow(), f);
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a + b)
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        self.elementwise_op_inplace(other, |a, b| a + b);
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a - b)
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        self.elementwise_op_inplace(other, |a, b| a - b);
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a * b)
    }
}

impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        self.elementwise_op_inplace(other, |a, b| a * b);
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a / b)
    }
}

impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        self.elementwise_op_inplace(other, |a, b| a / b);
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn add_elementwise() {
        let a = Tensor::new(&[3], &[1.0, 2.0, 3.0]);
        let b = Tensor::new(&[3], &[4.0, 5.0, 6.0]);
        let c = &a + &b;
        assert_eq!(&*c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_elementwise_with_broadcast() {
        let a = Tensor::new(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(&[3, 2], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let c = &a + &b;
        assert_eq!(&*c.data(), &[5.0, 6.0, 8.0, 9.0, 11.0, 12.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0]);
    }

    #[test]
    fn sub_elementwise() {
        let a = Tensor::new(&[3], &[4.0, 5.0, 6.0]);
        let b = Tensor::new(&[3], &[1.0, 2.0, 3.0]);
        let c = &a - &b;
        assert_eq!(&*c.data(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn mul_elementwise() {
        let a = Tensor::new(&[3], &[2.0, 3.0, 4.0]);
        let b = Tensor::new(&[3], &[2.0, 2.0, 2.0]);
        let c = &a * &b;
        assert_eq!(&*c.data(), &[4.0, 6.0, 8.0]);
    }

    #[test]
    fn div_elementwise() {
        let a = Tensor::new(&[3], &[6.0, 8.0, 9.0]);
        let b = Tensor::new(&[3], &[2.0, 4.0, 3.0]);
        let c = &a / &b;
        assert_eq!(&*c.data(), &[3.0, 2.0, 3.0]);
    }

    #[test]
    fn add_assign() {
        let mut a = Tensor::new(&[3], &[1.0, 2.0, 3.0]);
        let b = Tensor::new(&[3], &[1.0, 1.0, 1.0]);
        a += &b;
        assert_eq!(&*a.data(), &[2.0, 3.0, 4.0]);
    }
}
