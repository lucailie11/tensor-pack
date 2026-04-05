use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn matmul(&self, other: Tensor) -> Tensor {
        let raw: RawTensor = self.raw.borrow().matmul(&other.raw.borrow());
        Tensor::from_raw(raw, None)
    }
}

