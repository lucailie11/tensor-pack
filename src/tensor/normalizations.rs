use super::Tensor;
use crate::grad::BackpropOp;
use crate::rawtensor::RawTensor;

// Reduction operations along a single axis
// Delegates data logic to RawTensor and autograd logic to grad/ (not fully supported yet)
//
// Defined operations
// - softmax


impl Tensor {
    pub fn softmax(&self, axis: usize) -> Tensor {
        let raw: RawTensor = self.raw.softmax_axis(axis);
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Softmax(axis))
    }
}

