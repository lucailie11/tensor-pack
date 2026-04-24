use super::Tensor;
use crate::rawtensor::RawTensor;

// Delegate all ops to RawTensor 
// No gradient support so far

impl Tensor {
    pub fn dot(&self, other: &Tensor) -> Tensor {
        let raw: RawTensor = self.raw.dot(&other.raw);
        Tensor::no_grad_tensor(raw)
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let raw: RawTensor = self.raw.matmul(&other.raw);
        Tensor::no_grad_tensor(raw)
    }
}

