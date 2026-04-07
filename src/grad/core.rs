use std::rc::Rc;
use std::cell::RefCell;

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
        self.requires_grad || self.op != BackpropOp::None
    }
}

impl Tensor {
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if let Some(tensor) = Rc::get_mut(&mut self.0) {
            tensor.requires_grad = requires_grad;
        } else {
            panic!("Can't reshape non-leaf tensor");
        }
    }

    pub fn backprop(&self) {
        match self.op {
            BackpropOp::None => {},
            BackpropOp::AddTensor => { add_tensor_backprop(self, &self.inputs[0], &self.inputs[1]) },
            BackpropOp::SubTensor => { sub_tensor_backprop(self, &self.inputs[0], &self.inputs[1]) },
            BackpropOp::MulTensor => { mul_tensor_backprop(self, &self.inputs[0], &self.inputs[1]) },
            BackpropOp::DivTensor => { div_tensor_backprop(self, &self.inputs[0], &self.inputs[1]) },
        }
    }

    pub fn tensor_grad(raw: RawTensor, inputs: Box<[Tensor]>, op: BackpropOp) -> Tensor {
        Tensor::from_inner( 
            TensorInner {
                raw,
                grad: RefCell::new(None),
                op: if inputs.iter().any(|x| x.requires_computing_grad()) { op } else { BackpropOp::None },
                inputs,
                requires_grad: false,
            }
        )
    }
}
