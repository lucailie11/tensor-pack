use crate::Tensor;
use std::ops::Neg;

impl Tensor {
    fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.data.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn exp(&self) -> Tensor {
        self.map(f64::exp)
    }
    pub fn ln(&self) -> Tensor {
        self.map(f64::ln)
    }
    pub fn sqrt(&self) -> Tensor {
        self.map(f64::sqrt)
    }
    pub fn abs(&self) -> Tensor {
        self.map(f64::abs)
    }

    pub fn exp_inplace(&mut self) {
        self.map_inplace(f64::exp)
    }
    pub fn ln_inplace(&mut self) {
        self.map_inplace(f64::ln)
    }
    pub fn sqrt_inplace(&mut self) {
        self.map_inplace(f64::sqrt)
    }
    pub fn abs_inplace(&mut self) {
        self.map_inplace(f64::abs)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self * (-1.0)
    }
}
