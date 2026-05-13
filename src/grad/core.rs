use super::binary::{add_tensor_backprop, sub_tensor_backprop, mul_tensor_backprop, div_tensor_backprop};
use super::linalg::{dot_backprop, matmul_backprop};
use super::reductions::{mean_backprop, sum_backprop};
use super::normalizations::softmax_backprop;
use super::structure::{expand_backprop, squeeze_backprop, transpose_backprop, unsqueeze_backprop};
use super::scalar::{add_scalar_backprop, sub_scalar_backprop, mul_scalar_backprop, div_scalar_backprop};
use super::unary::{exp_backprop, ln_backprop, sqrt_backprop, abs_backprop, tanh_backprop, sigmoid_backprop, relu_backprop};
use crate::grad::structure::reshape_backprop;
use crate::rawtensor::RawTensor;
use crate::tensor::core::TensorInner;
use crate::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone, Copy, PartialEq, Debug)]
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

    Sum(usize),
    Mean(usize),

    Dot,
    Matmul,

    Softmax(usize),

    Reshape,
    Transpose,
    Expand,
    Squeeze(usize),
    Unsqueeze(usize),
}

impl Tensor {
    pub(super) fn tracks_grad(&self) -> bool {
        self.requires_grad || self.op != BackpropOp::None
    }

    pub fn requires_grad(mut self) -> Tensor {
        let inner = Rc::get_mut(&mut self.0).expect("can't set requires_grad on a shared tensor");
        inner.requires_grad = true; 
        self
    }

    pub(super) fn backprop(&self) {
        for input in self.inputs.iter() {
            if input.grad.borrow().is_none() && input.tracks_grad() {
                *input.grad.borrow_mut() = Some(RawTensor::zeros(input.shape()));
            }
        }

        match self.op {
            BackpropOp::None => {},

            // Tensor Tensor ops
            BackpropOp::AddTensor => { add_tensor_backprop(self, &self.inputs[0], &self.inputs[1]); }
            BackpropOp::SubTensor => { sub_tensor_backprop(self, &self.inputs[0], &self.inputs[1]); }
            BackpropOp::MulTensor => { mul_tensor_backprop(self, &self.inputs[0], &self.inputs[1]); }
            BackpropOp::DivTensor => { div_tensor_backprop(self, &self.inputs[0], &self.inputs[1]); }

            // Scalar Tensor ops
            BackpropOp::AddScalar => { add_scalar_backprop(self, &self.inputs[0]); }
            BackpropOp::SubScalar => { sub_scalar_backprop(self, &self.inputs[0]); }
            BackpropOp::MulScalar(scalar) => { mul_scalar_backprop(self, &self.inputs[0], scalar); }
            BackpropOp::DivScalar(scalar) => { div_scalar_backprop(self, &self.inputs[0], scalar); }

            // Unary ops
            BackpropOp::Exp     => { exp_backprop(self, &self.inputs[0]); }
            BackpropOp::Ln      => { ln_backprop(self, &self.inputs[0]); }
            BackpropOp::Sqrt    => { sqrt_backprop(self, &self.inputs[0]); }
            BackpropOp::Abs     => { abs_backprop(self, &self.inputs[0]); }
            BackpropOp::Tanh    => { tanh_backprop(self, &self.inputs[0]); }
            BackpropOp::Sigmoid => { sigmoid_backprop(self, &self.inputs[0]); }
            BackpropOp::Relu    => { relu_backprop(self, &self.inputs[0]); }

            // Reduction ops
            BackpropOp::Sum(axis)  => { sum_backprop(self, &self.inputs[0], axis); }
            BackpropOp::Mean(axis) => { mean_backprop(self, &self.inputs[0], axis); }

            // Linalg ops
            BackpropOp::Dot    => { dot_backprop(self, &self.inputs[0], &self.inputs[1]); }
            BackpropOp::Matmul => { matmul_backprop(self, &self.inputs[0], &self.inputs[1]); }

            // Normalization ops
            BackpropOp::Softmax(axis) => { softmax_backprop(self, &self.inputs[0], axis); }     // Does nothing
            
            // Structure ops
            BackpropOp::Reshape   => { reshape_backprop(self, &self.inputs[0]); }
            BackpropOp::Transpose => { transpose_backprop(self, &self.inputs[0]); }
            BackpropOp::Expand    => { expand_backprop(self, &self.inputs[0]); }
            BackpropOp::Squeeze(axis)   => { squeeze_backprop(self, &self.inputs[0], axis); }
            BackpropOp::Unsqueeze(axis) => { unsqueeze_backprop(self, &self.inputs[0], axis); }
        }
    }

    pub(crate) fn autograd_tensor(raw: RawTensor, inputs: Box<[Tensor]>, op: BackpropOp) -> Tensor {
        Tensor::from_inner( 
            TensorInner {
                raw,
                grad: RefCell::new(None),
                op: if inputs.iter().any(|x| x.tracks_grad()) { op } else { BackpropOp::None },
                inputs,
                requires_grad: false,
            }
        )
    }
}
