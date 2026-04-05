use crate::rawtensor::RawTensor;
use super::Tensor;
use crate::utils::stride::softmax;

impl Tensor {
    pub fn apply_axis(&self, axis: usize, f: impl Fn(&mut [f64], usize)) -> Tensor {
        let raw: RawTensor = self.raw.borrow().apply_axis(axis, f);
        Tensor::from_raw(raw, None)
    }

    pub fn apply_axis_inplace(&mut self, axis: usize, f: impl Fn(&mut [f64], usize)) {
        self.raw.borrow_mut().apply_axis_inplace(axis, f);
    }

    pub fn softmax(&self, axis: usize) -> Tensor {
        self.apply_axis(axis, softmax)
    }
 
    pub fn softmax_inplace(&mut self, axis: usize) {
        self.apply_axis_inplace(axis, softmax);
    }
}

