use super::Tensor;
use crate::utils::stride::{sum, mean, var, std_dev};

impl Tensor {
    pub fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64], usize) -> f64) -> Tensor {
        let raw = self.raw.borrow().reduce_axis(axis, f);
        Tensor::from_raw(raw, None)
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, sum)
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, mean)
    }

    pub fn var_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, var)
    }

    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, std_dev)
    }
}

