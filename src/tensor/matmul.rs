use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn matmul(&self, other: Tensor) -> Tensor {
        let raw: RawTensor = self.raw.matmul(&other.raw);
        Tensor::no_grad_tensor(raw)
    }
}

