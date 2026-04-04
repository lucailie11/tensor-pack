use super::RawTensor;
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
//   &RawTensor + &RawTensor  -> RawTensor
//    RawTensor += &RawTensor
//   &RawTensor - &RawTensor  -> RawTensor
//    RawTensor -= &RawTensor
//   &RawTensor * &RawTensor  -> RawTensor
//    RawTensor *= &RawTensor
//   &RawTensor / &RawTensor  -> RawTensor
//    RawTensor /= &RawTensor

impl RawTensor {
    // Panics if the shapes are incompatible for broadcasting.
    pub fn elementwise_op(&self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) -> RawTensor {
        let out_shape = get_broadcast_shape(&self.shape, &other.shape);
        let out_len: usize = out_shape.iter().product();
        let new_data: Vec<f64> = (0..out_len)
            .map(|i| f(
                self.data[get_broadcast_index(i, &self.shape, &out_shape)],
                other.data[get_broadcast_index(i, &other.shape, &out_shape)],
            ))
            .collect();

        RawTensor {
            shape: out_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // Panics if self doesn't already have the broadcast output shape (other is broadcast into self).
    pub fn elementwise_op_inplace(&mut self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) {
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
