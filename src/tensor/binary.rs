use crate::Tensor;
use crate::tensor::broadcasting::{get_broadcast_index, get_broadcast_shape};
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

// Element-wise binary operations between two Tensors. Supports broadcasting.
// For _inplace ops, self must already be the output shape (other broadcasts into self).
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
    // Panics if the Tensors are not broadcastable
    pub fn elementwise_op(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Tensor {
        let out_shape = get_broadcast_shape(&self.shape, &other.shape);
        let out_len: usize = out_shape.iter().product();
        let new_data: Vec<f64> = (0..out_len)
            .map(|i| f(
                self.data[get_broadcast_index(i, &self.shape, &out_shape)],
                other.data[get_broadcast_index(i, &other.shape, &out_shape)],
            ))
            .collect();

        Tensor {
            shape: out_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // Panics if self is not already the broadcast output shape (other must broadcast into self).
    pub fn elementwise_op_inplace(&mut self, other: &Tensor, f: impl Fn(f64, f64) -> f64) {
        assert_eq!(
            get_broadcast_shape(&self.shape, &other.shape).as_slice(),
            &*self.shape,
            "in-place broadcasting requires self to already be the output shape"
        );
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i], other.data[get_broadcast_index(i, &other.shape, &self.shape)]);
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
