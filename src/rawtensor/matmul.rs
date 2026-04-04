use super::RawTensor;

// Matrix multiplication: shapes [m,k] × [k,n] → [m,n]. Both tensors must be 2D.
// Use tensor.matmul(other) instead of * (which is element-wise, like NumPy/PyTorch).

impl RawTensor {
    pub fn matmul(&self, other: &RawTensor) -> RawTensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "shape mismatch: [m,k] × [k,n] required");

        let mut new_data: Vec<f64> = vec![0.0; self.shape[0] * other.shape[1]];
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                for k in 0..self.shape[1] {
                    new_data[i * other.shape[1] + j] +=
                        self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
            }
        }

        RawTensor {
            shape: vec![self.shape[0], other.shape[1]].into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }
}
