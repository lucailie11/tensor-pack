use crate::{Tensor, rawtensor::RawTensor, tensor::core::TensorInner};
use super::binary::{add_tensor_backprop, sub_tensor_backprop, div_tensor_backprop, mul_tensor_backprop};
                                                               
#[derive(Clone, Copy, PartialEq)]
pub enum BackpropOp {
    None,                                                         
    AddTensor,                                                   
    SubTensor,
    MulTensor,
    DivTensor,
}

impl Tensor {
    pub fn requires_computing_grad(&self) -> bool {
        self.borrow().requires_grad || self.borrow().backprop_op != BackpropOp::None
    }

    pub fn backprop(&self) {
        match self.borrow().backprop_op {
            BackpropOp::None => {},
            BackpropOp::AddTensor => { add_tensor_backprop(&self, &self.borrow().inputs) },
            BackpropOp::SubTensor => { sub_tensor_backprop(&self, &self.borrow().inputs) },
            BackpropOp::MulTensor => { mul_tensor_backprop(&self, &self.borrow().inputs) },
            BackpropOp::DivTensor => { div_tensor_backprop(&self, &self.borrow().inputs) },
        }
    }

    pub fn tensor_grad(raw: RawTensor, inputs: Box<[Tensor]>, backprop_op: BackpropOp) -> Tensor {
        Tensor::from_inner(
            TensorInner {
                raw,
                grad: None,
                backprop_op: if inputs.iter().any(|x| x.requires_computing_grad()) { backprop_op } else { BackpropOp::None },
                inputs,
                requires_grad: false,
            }
        )
    }
}
