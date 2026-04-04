use super::RawTensor;
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

impl RawTensor {
    pub fn map(&self, f: impl Fn(f64) -> f64) -> RawTensor {
        RawTensor {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .map(|x| f(*x))
                .collect::<Vec<f64>>()
                .into_boxed_slice(),
        }
    }

    pub fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.data.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn exp(&self) -> RawTensor     { self.map(f64::exp)  }
    pub fn ln(&self) -> RawTensor      { self.map(f64::ln)   }
    pub fn sqrt(&self) -> RawTensor    { self.map(f64::sqrt) }
    pub fn abs(&self) -> RawTensor     { self.map(f64::abs)  }
    pub fn tanh(&self) -> RawTensor    { self.map(f64::tanh) }
    pub fn sigmoid(&self) -> RawTensor { self.map(sigmoid)   }
    pub fn relu(&self) -> RawTensor    { self.map(relu)      }

    pub fn exp_inplace(&mut self)     { self.map_inplace(f64::exp)  }
    pub fn ln_inplace(&mut self)      { self.map_inplace(f64::ln)   }
    pub fn sqrt_inplace(&mut self)    { self.map_inplace(f64::sqrt) }
    pub fn abs_inplace(&mut self)     { self.map_inplace(f64::abs)  }
    pub fn tanh_inplace(&mut self)    { self.map_inplace(f64::tanh) }
    pub fn sigmoid_inplace(&mut self) { self.map_inplace(sigmoid)   }
    pub fn relu_inplace(&mut self)    { self.map_inplace(relu)      }
}

impl Neg for &RawTensor {
    type Output = RawTensor;

    fn neg(self) -> RawTensor {
        self.map(|x| -x)
    }
}
