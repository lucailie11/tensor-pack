use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn softmax(&self, axis: usize) -> Tensor {
        let raw: RawTensor = self.raw.softmax(axis);
        Tensor::no_grad_tensor(raw)
    }
 
    // pub fn softmax_inplace(&mut self, axis: usize) {
    //     self.raw.softmax_inplace(axis);
    // }
}

