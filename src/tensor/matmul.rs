use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn matmul(&self, other: Tensor) -> Tensor {
        let raw: RawTensor = self.data.borrow().matmul(&other.data.borrow());
        Tensor::new_tensor(raw, None)
    }
}
