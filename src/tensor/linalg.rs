use super::Tensor;
use crate::rawtensor::RawTensor;
use crate::grad::BackpropOp;

// Dot product and matrix multiplication between two Tensor
// Delegates data logic to RawTensor and autograd logic to grad/

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

