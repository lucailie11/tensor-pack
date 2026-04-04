use crate::rawtensor::RawTensor;
use super::Tensor;
use crate::utils::stride::softmax;

impl Tensor {
    pub fn apply_axis(&self, axis: usize, f: impl Fn(&mut [f64], usize)) -> Tensor {
        let raw: RawTensor = self.raw.borrow().apply_axis(axis, f);
        Tensor::new_tensor(raw, None)
    }

    pub fn apply_axis_inplace(&mut self, axis: usize, f: impl Fn(&mut [f64], usize)) {
        self.raw.borrow_mut().apply_axis_inplace(axis, f);
    }

    pub fn softmax(&self, axis: usize) -> Tensor {
        self.apply_axis(axis, softmax)
    }
 
    pub fn softmax_inplace(&mut self, axis: usize) {
        self.apply_axis_inplace(axis, softmax);
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn softmax_sums_to_one() {
        let a = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.softmax(0);
        let sum: f64 = b.data().iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn softmax_2d_rows_sum_to_one() {
        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let b = a.softmax(1);
        let data = b.raw.borrow();
        let row0: f64 = data.data()[0..3].iter().sum();
        let row1: f64 = data.data()[3..6].iter().sum();
        assert!(approx_eq(row0, 1.0));
        assert!(approx_eq(row1, 1.0));
    }
}
