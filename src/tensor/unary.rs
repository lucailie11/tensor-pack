use super::Tensor;
use crate::grad::BackpropOp;

use std::ops::Neg;

// Unary operations on Tensors
// Delegates data logic to RawTensor and autograd logic to grad/ 
// Assign operations replace the left-hand side with the result (not in-place)
//
// Defined operations:
//   general map (has no grad)
//   exp, ln, sqrt, abs, tanh, sigmoid, relu
//   -&RawTensor — negation

impl Tensor {
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let raw = self.raw.map(f);
        Tensor::no_grad_tensor(raw)
    }

    pub fn exp(&self) -> Tensor {
        let raw = self.raw.exp();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Exp)
    }
    pub fn ln(&self) -> Tensor {
        let raw = self.raw.ln();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Ln)
    }
    pub fn sqrt(&self) -> Tensor {
        let raw = self.raw.sqrt();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Sqrt)
    }
    pub fn abs(&self) -> Tensor {
        let raw = self.raw.abs();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Abs)
    }
    pub fn tanh(&self) -> Tensor {
        let raw = self.raw.tanh();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Tanh)
    }
    pub fn sigmoid(&self) -> Tensor {
        let raw = self.raw.sigmoid();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Sigmoid)
    }
    pub fn relu(&self) -> Tensor {
        let raw = self.raw.relu();
        Tensor::autograd_tensor(raw, Box::from([self.clone()]), BackpropOp::Relu)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        0.0 - self
    }
}

