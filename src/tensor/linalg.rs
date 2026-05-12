use super::Tensor;
use crate::rawtensor::RawTensor;
use crate::grad::BackpropOp;

// Delegate all ops to RawTensor 
// No gradient support so far

impl Tensor {
    pub fn dot(&self, other: &Tensor) -> Tensor {
        let raw: RawTensor = self.raw.dot(&other.raw);
        Tensor::autograd_tensor(raw, Box::from([self.clone(), other.clone()]), BackpropOp::Dot)
        
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let raw: RawTensor = self.raw.matmul(&other.raw);
        Tensor::autograd_tensor(raw, Box::from([self.clone(), other.clone()]), BackpropOp::Matmul)
    }
}

