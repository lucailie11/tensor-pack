use super::Tensor;
use super::broadcasting::{get_broadcast_index, get_broadcast_shape};
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::{Mul, MulAssign};
use std::ops::{Div, DivAssign};

// Binary element-wise operations on two Tensors with broadcasting support.
// In-place variants require self to already hold the output shape (other is broadcast into it)
//
// TODO: optimizing broadcasting index lookup
//
// The core building blocks are elementwise_op and elementwise_op_inplace.
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
    // Panics if the shapes are incompatible for broadcasting.
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

    // Panics if self doesn't already have the broadcast output shape (other is broadcast into self).
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
