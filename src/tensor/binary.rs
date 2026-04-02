use crate::Tensor;
use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};

// Element-wise binary operations between two Tensors.
// Broadcasting is not yet implemented; shapes must match exactly (same total element count).
//
// elementwise_op / elementwise_op_inplace are the core primitives.
// + and - delegate to these. Matmul (&Tensor * &Tensor) lives in matmul.rs.
// Scalar ops (&Tensor op f64) live in scalar.rs.
//
// Defined operations:
//   &Tensor + &Tensor  -> Tensor
//    Tensor += &Tensor
//   &Tensor - &Tensor  -> Tensor
//    Tensor -= &Tensor

impl Tensor {
    // Applies f element-wise to self and other, returning a new Tensor with self's shape.
    // Panics if the two tensors have different total element counts.
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

    // Applies f element-wise in place. Panics if lengths differ.
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
