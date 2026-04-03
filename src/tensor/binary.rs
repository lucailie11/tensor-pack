use crate::Tensor;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

// Element-wise binary operations between two Tensors.
// Broadcasting is not yet implemented; shapes must match exactly (same total element count).
//
// elementwise_op / elementwise_op_inplace are the core primitives.
//
// Defined operations:
//   &Tensor + &Tensor  -> Tensor
//    Tensor += &Tensor
//   &Tensor - &Tensor  -> Tensor
//    Tensor -= &Tensor
//   &Tensor * &Tensor  -> Tensor
//    Tensor *= &Tensor
//   &Tensor / &Tensor  -> Tensor
//    Tensor /= &Tensor

impl Tensor {
    pub fn elementwise_op(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Tensor {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch");

        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| f(*a, *b))
            .collect();

        Tensor {
            shape: self.shape.clone(),
            data: new_data.into_boxed_slice(),
        }
    }

    pub fn elementwise_op_inplace(&mut self, other: &Tensor, f: impl Fn(f64, f64) -> f64) {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch");

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = f(*a, *b)
        }
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
