use super::RawTensor;
use std::rc::Rc;

// Matrix multiplication: shapes [m, k] x [k, n] -> [m, n]. Both tensors must be 2D.
// Use tensor.matmul(other) instead of * (which is element-wise)

impl RawTensor {
    pub fn dot(&self, other: &RawTensor) -> f64 {
        assert_eq!(self.shape.len(), 1, "dot requires 1D tensors");
        assert_eq!(other.shape.len(), 1, "dot requires 1D tensors");
        assert_eq!(self.shape[0], other.shape[0], "shape mismatch: [m, k] x [k, n] required");

        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn matmul(&self, other: &RawTensor) -> RawTensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "shape mismatch: [m, k] x [k, n] required");

        let mut new_data: Box<[f64]> = Box::from(vec![0.0; self.shape[0] * other.shape[1]]);
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

        RawTensor::from_box(&[self.shape[0], other.shape[1]], new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_standard() {
        let a = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let c = a.dot(&b);
        assert_eq!(c, 30.0);
    }

    #[test]
    fn matmul_2x2() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_output_shape() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        let b = RawTensor::from_slice(&[3, 4], &[1.0; 12]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn matmul_values() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[2, 2], &[2.0, 0.0, 1.0, 3.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data(), &[4.0, 6.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic]
    fn wrong_shape() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = RawTensor::from_slice(&[2, 2], &[2.0, 0.0, 1.0, 3.0]);
        let _c = a.matmul(&b);
    }

    #[test]
    #[should_panic]
    fn not_matrix() {
        let a = RawTensor::from_slice(&[6], &[1.0, 2.0, 3.0, 4.0]);
        let b = RawTensor::from_slice(&[1, 2, 2], &[2.0, 0.0, 1.0, 3.0]);
        let _c = a.matmul(&b);
    }

}
