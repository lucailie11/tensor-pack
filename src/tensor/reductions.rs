use crate::grad::BackpropOp;

use super::Tensor;

// Delegate all ops to RawTensor 
// No gradient support so far

impl Tensor {
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.sum_axis(axis);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Sum(axis))
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.mean_axis(axis);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Mean(axis))
    }

    pub fn var_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.var_axis(axis); 
        Tensor::no_grad_tensor(raw) 
    }

    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.std_dev_axis(axis);
        Tensor::no_grad_tensor(raw)
    }
}

