use super::Tensor;

impl Tensor {
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let raw = self.borrow().raw.sum_axis(axis);
        Tensor::from_raw(raw, None)
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let raw = self.borrow().raw.mean_axis(axis);
        Tensor::from_raw(raw, None)
    }

    pub fn var_axis(&self, axis: usize) -> Tensor {
        let raw = self.borrow().raw.var_axis(axis); 
        Tensor::from_raw(raw, None) 
    }

    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        let raw = self.borrow().raw.std_dev_axis(axis);
        Tensor::from_raw(raw, None)
    }
}

