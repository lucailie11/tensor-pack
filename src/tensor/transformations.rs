use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn softmax(&self, axis: usize) -> Tensor {
        let raw: RawTensor = self.borrow().raw.softmax(axis);
        Tensor::from_raw(raw, None)
    }
 
    pub fn softmax_inplace(&mut self, axis: usize) {
        self.borrow_mut().raw.softmax_inplace(axis);
    }
}

