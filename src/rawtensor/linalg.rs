use super::RawTensor;

use std::rc::Rc;

// Both dot and matmul return a RawTensor with a fresh data allocation
impl RawTensor {
    // Dot product: Both tensors must be 1D and have the same shape
    pub fn dot(&self, other: &RawTensor) -> RawTensor {
        assert_eq!(self.shape.len(), 1, "dot requires 1D tensors");
        assert_eq!(other.shape.len(), 1, "dot requires 1D tensors");
        assert_eq!(self.shape[0], other.shape[0], "shape mismatch");

        let p: f64 = self.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        RawTensor::from_scalar(p)
    }

    // Matrix multiplication: shapes [m, k] x [k, n] -> [m, n]. Both tensors must be 2D
    pub fn matmul(&self, other: &RawTensor) -> RawTensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "shape mismatch: [m, k] x [k, n] required");

        let mut new_data: Vec<f64> = vec![0.0; self.shape[0] * other.shape[1]];
        let a: Rc<[f64]> = self.contiguous_data();
        let b: Rc<[f64]> = other.contiguous_data();
        for i in 0..self.shape[0] {
            for k in 0..self.shape[1] {
                for j in 0..other.shape[1] {
                    new_data[i * other.shape[1] + j] +=
                        a[i * self.shape[1] + k] * b[k * other.shape[1] + j];
                }
            }
        }

        RawTensor::from_vec(&[self.shape[0], other.shape[1]], new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_standard() {
        let a = RawTensor::linspace(1.0, 4.0, 4);
        let b = RawTensor::linspace(1.0, 4.0, 4);
        let c = a.dot(&b);
        assert_eq!(*c.shape, []);
        assert_eq!(*c.contiguous_data(), [30.0]);
    }

    #[test]
    fn dot_non_contiguous() {
        let a = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_scalar(2.0).expand(&[4]);
        assert!(!b.is_contiguous());
        assert_eq!(*a.dot(&b).contiguous_data(), [20.0]);
    }

    #[test]
    fn matmul_id() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let c = a.matmul(&b);
        assert_eq!(*c.shape, [2, 2]);
        assert_eq!(*c.contiguous_data(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_values() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[2, 2], &[2.0, 0.0, 1.0, 3.0]);
        let c = a.matmul(&b);
        assert_eq!(*c.shape, [2, 2]);
        assert_eq!(*c.contiguous_data(), [4.0, 6.0, 10.0, 12.0]);
    }

    #[test]
    fn matmul_5x10_10x7() {
        let a = RawTensor::linspace(1.0, 50.0, 50).reshape(&[5, 10]);
        let b = RawTensor::full(&[10, 7], 1.0);
        let c = a.matmul(&b);
        assert_eq!(*c.shape, [5, 7]);
        let expected: Vec<f64> = [55.0, 155.0, 255.0, 355.0, 455.0]
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, 7))
            .collect();
        assert_eq!(*c.contiguous_data(), *expected);
    }

    #[test]
    fn matmul_transposed_arg() {
        let a = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let b = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).transpose(&[1, 0]);
        assert!(!b.is_contiguous());
        let c = a.matmul(&b);
        assert_eq!(*c.shape, [2, 2]);
        assert_eq!(*c.contiguous_data(), [14.0, 32.0, 32.0, 77.0]);
    }

    #[test]
    #[should_panic]
    fn wrong_shape() {
        let a = RawTensor::zeros(&[2, 6]);
        let b = RawTensor::zeros(&[7, 2]);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic]
    fn not_matrix_1() {
        let a = RawTensor::zeros(&[4]);
        let b = RawTensor::zeros(&[4, 2]);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic]
    fn not_matrix_2() {
        let a = RawTensor::zeros(&[2, 4]);
        let b = RawTensor::zeros(&[4]);
        let _ = a.matmul(&b);
    }
}
