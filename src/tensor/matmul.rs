use crate::rawtensor::RawTensor;
use super::Tensor;

impl Tensor {
    pub fn matmul(&self, other: Tensor) -> Tensor {
        let raw: RawTensor = self.data.borrow().matmul(&other.data.borrow());
        Tensor::new_tensor(raw, None)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn matmul_2x2() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let c = a.matmul(b);
        assert_eq!(&*c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_output_shape() {
        let a = Tensor::new(&[2, 3], &[1.0; 6]);
        let b = Tensor::new(&[3, 4], &[1.0; 12]);
        let c = a.matmul(b);
        assert_eq!(&*c.shape(), &[2, 4]);
    }

    #[test]
    fn matmul_values() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(&[2, 2], &[2.0, 0.0, 1.0, 3.0]);
        let c = a.matmul(b);
        assert_eq!(&*c.data(), &[4.0, 6.0, 10.0, 12.0]);
    }
}
