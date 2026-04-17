use super::Tensor;
use std::ops::Neg;
use crate::grad::BackpropOp;

// map / map_inplace are the core primitives: apply a closure to every element,
// either returning a new RawTensor or mutating in place.
//
// Scalar arithmetic (+f64, -f64, …) also builds on map but lives in scalar.rs.
// Unary negation (-&RawTensor) is here since it takes no second operand.
//
// Defined operations:
//   exp, ln, sqrt, abs, tanh, sigmoid, relu  (and *_inplace variants)
//   -&RawTensor — negation

impl Tensor {
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let raw = self.0.raw.map(f);
        Tensor::from_raw(raw, None)
    }

    // pub fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
    //     self.raw.map_inplace(f);
    // }

    pub fn exp(&self) -> Tensor {
        let raw = self.raw.exp();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Exp)
    }
    pub fn ln(&self) -> Tensor {
        let raw = self.raw.ln();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Ln)
    }
    pub fn sqrt(&self) -> Tensor {
        let raw = self.raw.sqrt();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Sqrt)
    }
    pub fn abs(&self) -> Tensor {
        let raw = self.raw.abs();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Abs)
    }
    pub fn tanh(&self) -> Tensor {
        let raw = self.raw.tanh();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Tanh)
    }
    pub fn sigmoid(&self) -> Tensor {
        let raw = self.raw.sigmoid();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Sigmoid)
    }
    pub fn relu(&self) -> Tensor {
        let raw = self.raw.relu();
        Tensor::tensor_grad(raw, Box::from([self.clone()]), BackpropOp::Relu)
    }

    // pub fn exp_inplace(&mut self)     { self.map_inplace(f64::exp)  }
    // pub fn ln_inplace(&mut self)      { self.map_inplace(f64::ln)   }
    // pub fn sqrt_inplace(&mut self)    { self.map_inplace(f64::sqrt) }
    // pub fn abs_inplace(&mut self)     { self.map_inplace(f64::abs)  }
    // pub fn tanh_inplace(&mut self)    { self.map_inplace(f64::tanh) }
    // pub fn sigmoid_inplace(&mut self) { self.map_inplace(sigmoid)   }
    // pub fn relu_inplace(&mut self)    { self.map_inplace(relu)      }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        0.0 - self
    }
}

