use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn matmul(&self, other: Tensor) -> Tensor {
        let raw: RawTensor = self.borrow().raw.matmul(&other.borrow().raw);
        Tensor::from_raw(raw, None)
    }
}

