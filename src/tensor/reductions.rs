use super::Tensor;

impl Tensor {
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.sum_axis(axis);
        Tensor::no_grad_tensor(raw)
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.mean_axis(axis);
        Tensor::no_grad_tensor(raw)
    }

    pub fn var_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.var_axis(axis); 
        Tensor::no_grad_tensor(raw) 
    }

    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        let raw = self.raw.std_dev_axis(axis);
        Tensor::no_grad_tensor(raw)
    }
}

