use crate::grad::BackpropOp;
use super::Tensor;
use std::ops::Index;

// Structure operations
// Delegates data logic to RawTensor and autograd logic to grad/
//
// Defined operations:
// - reshape (only for contiguous Tensor)
// - transpose 
// - expand
// - squeeze
// - unsqueeze

impl Tensor {
    // Returns a new Tensor with a new shape. Panics if tensor is not contiguous
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let raw = self.raw.reshape(new_shape);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Reshape)
    }

    // Returns a new RawTensor with dimensions permuted (new dim_i comes from old dim_perm[i])
    pub fn transpose(&self, perm: &[usize]) -> Tensor {
        let raw = self.raw.transpose(perm);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Transpose)
    }

    // Expands self to new_shape. Panics if self is not broadcastable to new_shape
    pub fn expand(&self, new_shape: &[usize]) -> Tensor {
        let raw = self.raw.expand(new_shape);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Expand)
    }

    // Removes a single size-1 axis
    pub fn squeeze(&self, axis: usize) -> Tensor {
        let raw = self.raw.squeeze(axis);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Squeeze(axis))
    }

    // Inserts a size-1 axis at the given position
    pub fn unsqueeze_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.unsqueeze(axis);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Unsqueeze(axis))
    }

}

impl Tensor {
    // Returns None if indices are out of bounds or wrong number of dims.
    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        self.raw.get(indices)
    }
}

// Panics if indices are out of bounds or wrong number of dims.
impl Index<&[usize]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize]) -> &f64 {
        &self.raw[indices]
    }
}
