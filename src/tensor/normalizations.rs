use super::Tensor;
use crate::grad::BackpropOp;
use crate::rawtensor::RawTensor;

// Delegate all ops to RawTensor 
// No gradient support so far

impl Tensor {
    pub fn softmax(&self, axis: usize) -> Tensor {
        let raw: RawTensor = self.raw.softmax_axis(axis);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Softmax(axis))
    }
}

