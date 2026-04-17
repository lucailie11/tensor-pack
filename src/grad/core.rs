use std::rc::Rc;
use std::cell::RefCell;

use crate::{Tensor, rawtensor::RawTensor, tensor::core::TensorInner};
use super::binary::{add_tensor_backprop, sub_tensor_backprop, mul_tensor_backprop, div_tensor_backprop};
use super::scalar::{add_scalar_backprop, sub_scalar_backprop, mul_scalar_backprop, div_scalar_backprop};
use super::unary::{exp_backprop, ln_backprop, sqrt_backprop, abs_backprop, tanh_backprop, sigmoid_backprop, relu_backprop};

#[derive(Clone, Copy, PartialEq)]
pub enum BackpropOp {
    None,
    AddTensor,
    SubTensor,
    MulTensor,
    DivTensor,
    AddScalar,
    SubScalar,
    MulScalar(f64),
    DivScalar(f64),
    Exp,
    Ln,
    Sqrt,
    Abs,
    Tanh,
    Sigmoid,
    Relu,
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
            panic!("Can't change requires_grad on a non-leaf tesnor");
        }
    }

    pub fn backprop(&self) {
        for input in self.inputs.iter() {
            if input.grad.borrow().is_none() {
                *input.grad.borrow_mut() = Some(RawTensor::zeros(input.shape()));
            }
        }

        match self.op {
            BackpropOp::None => {},
            BackpropOp::AddTensor => {add_tensor_backprop(self, &self.inputs[0], &self.inputs[1]);}
            BackpropOp::SubTensor => {sub_tensor_backprop(self, &self.inputs[0], &self.inputs[1]);}
            BackpropOp::MulTensor => {mul_tensor_backprop(self, &self.inputs[0], &self.inputs[1]);}
            BackpropOp::DivTensor => {div_tensor_backprop(self, &self.inputs[0], &self.inputs[1]);}
            BackpropOp::AddScalar => {add_scalar_backprop(self, &self.inputs[0]);}
            BackpropOp::SubScalar => {sub_scalar_backprop(self, &self.inputs[0]);}
            BackpropOp::MulScalar(scalar) => {mul_scalar_backprop(self, &self.inputs[0], scalar);}
            BackpropOp::DivScalar(scalar) => {div_scalar_backprop(self, &self.inputs[0], scalar);}
            BackpropOp::Exp => {exp_backprop(self, &self.inputs[0]);}
            BackpropOp::Ln => {ln_backprop(self, &self.inputs[0]);}
            BackpropOp::Sqrt => {sqrt_backprop(self, &self.inputs[0]);}
            BackpropOp::Abs => {abs_backprop(self, &self.inputs[0]);}
            BackpropOp::Tanh => {tanh_backprop(self, &self.inputs[0]);}
            BackpropOp::Sigmoid => {sigmoid_backprop(self, &self.inputs[0]);}
            BackpropOp::Relu => {relu_backprop(self, &self.inputs[0]);}
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
