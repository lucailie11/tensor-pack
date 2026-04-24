use super::Tensor;

use std::ops::Index;

// Delegate all ops to RawTensor 
// No gradient support so far

impl Tensor {
    // Returns a new Tensor with a new shape. Panics if tensor is not contiguous
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let raw = self.raw.reshape(new_shape);
        Tensor::no_grad_tensor(raw)
    }

    // Returns a new Tensor with dimensions permuted according to perm
    pub fn transpose(&self, perm: &[usize]) -> Tensor {
        let raw = self.raw.transpose(perm);
        Tensor::no_grad_tensor(raw)
    }

    // Expands self to new_shape. Panics if self is not broadcastable to new_shape
    pub fn expand(&self, new_shape: &[usize]) -> Tensor {
        let raw = self.raw.expand(new_shape);
        Tensor::no_grad_tensor(raw)
    }

    // Removes a set of size-1 axes from the shape
    pub fn squeeze_axes(&self, axes: &[usize]) -> Tensor {
        let raw = self.raw.squeeze_axes(axes);
        Tensor::no_grad_tensor(raw)
    }

    // Removes a single size-1 axis
    pub fn squeeze_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.squeeze_axis(axis);
        Tensor::no_grad_tensor(raw)
    }

    // Removes all size-1 axes
    pub fn squeeze_all(&self) -> Tensor {
        let raw = self.raw.squeeze_all();
        Tensor::no_grad_tensor(raw)
    }

    // Inserts a size-1 axis at the given position
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        let raw = self.raw.unsqueeze(axis);
        Tensor::no_grad_tensor(raw)
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
