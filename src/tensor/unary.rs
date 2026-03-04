use crate::Tensor;
use std::ops::Neg;

impl Tensor {
    pub fn exp(&self) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a.exp()).collect();
        Tensor {
            shape: self.shape.to_vec(),
            data: new_data,
        }
    }

    pub fn exp_self(&mut self) {
        for a in self.data.iter_mut() {
            *a = a.exp();
        }
    }
}

impl Tensor {
    pub fn ln(&self) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a.ln()).collect();
        Tensor {
            shape: self.shape.to_vec(),
            data: new_data,
        }
    }

    pub fn ln_self(&mut self) {
        for a in self.data.iter_mut() {
            *a = a.ln();
        }
    }
}

impl Tensor {
    pub fn sqrt(&self) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a.sqrt()).collect();
        Tensor {
            shape: self.shape.to_vec(),
            data: new_data,
        }
    }

    pub fn sqrt_self(&mut self) {
        for a in self.data.iter_mut() {
            *a = a.sqrt();
        }
    }
}

impl Tensor {
    pub fn abs(&self) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|a| a.abs()).collect();
        Tensor {
            shape: self.shape.to_vec(),
            data: new_data,
        }
    }

    pub fn abs_self(&mut self) {
        for a in self.data.iter_mut() {
            *a = a.abs();
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self * (-1.0)
    }
}
