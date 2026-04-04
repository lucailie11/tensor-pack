use super::Tensor;
use crate::utils::stride::{sum, mean, var, std_dev};

impl Tensor {
    pub fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64], usize) -> f64) -> Tensor {
        let raw = self.raw.borrow().reduce_axis(axis, f);
        Tensor::new_tensor(raw, None)
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

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    #[test]
    fn sum_axis0() {
        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.sum_axis(0);
        assert_eq!(&*b.shape(), &[3]);
        assert_eq!(&*b.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis1() {
        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.sum_axis(1);
        assert_eq!(&*b.shape(), &[2]);
        assert_eq!(&*b.data(), &[6.0, 15.0]);
    }

    #[test]
    fn mean_axis0() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.mean_axis(0);
        assert!(b.data().iter().zip(&[2.0, 3.0]).all(|(&x, &y)| approx_eq(x, y)));
    }
}
