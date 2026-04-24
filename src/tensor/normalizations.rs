use super::Tensor;
use crate::rawtensor::RawTensor;

// Delegate all ops to RawTensor 
// No gradient support so far

impl Tensor {
    pub fn softmax(&self, axis: usize) -> Tensor {
        let raw: RawTensor = self.raw.softmax_axis(axis);
        Tensor::no_grad_tensor(raw)
    }
}

