use super::Tensor;
use std::ops::Neg;

// map / map_inplace are the core primitives: apply a closure to every element,
// either returning a new RawTensor or mutating in place.
//
// Scalar arithmetic (+f64, -f64, …) also builds on map but lives in scalar.rs.
// Unary negation (-&RawTensor) is here since it takes no second operand.
//
// Defined operations:
//   exp, ln, sqrt, abs, tanh, sigmoid, relu  (and *_inplace variants)
//   -&RawTensor — negation

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0.0 }
}

impl Tensor {
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let raw = self.borrow().raw.map(f);
        Tensor::from_raw(raw, None)
    }

    pub fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.borrow_mut().raw.map_inplace(f);
    }

    pub fn exp(&self) -> Tensor     { self.map(f64::exp)  }
    pub fn ln(&self) -> Tensor      { self.map(f64::ln)   }
    pub fn sqrt(&self) -> Tensor    { self.map(f64::sqrt) }
    pub fn abs(&self) -> Tensor     { self.map(f64::abs)  }
    pub fn tanh(&self) -> Tensor    { self.map(f64::tanh) }
    pub fn sigmoid(&self) -> Tensor { self.map(sigmoid)   }
    pub fn relu(&self) -> Tensor    { self.map(relu)      }

    pub fn exp_inplace(&mut self)     { self.map_inplace(f64::exp)  }
    pub fn ln_inplace(&mut self)      { self.map_inplace(f64::ln)   }
    pub fn sqrt_inplace(&mut self)    { self.map_inplace(f64::sqrt) }
    pub fn abs_inplace(&mut self)     { self.map_inplace(f64::abs)  }
    pub fn tanh_inplace(&mut self)    { self.map_inplace(f64::tanh) }
    pub fn sigmoid_inplace(&mut self) { self.map_inplace(sigmoid)   }
    pub fn relu_inplace(&mut self)    { self.map_inplace(relu)      }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.map(|x| -x)
    }
}

