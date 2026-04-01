use crate::Tensor;
use std::ops::{Mul, MulAssign};

// Tensor * Tensor: matrix multiplication (both tensors must be 2D, shapes [m,k] and [k,n] → [m,n]).
// This is NOT element-wise; use scalar multiplication for scaling.
//
// Tensor * f64 or f64 * Tensor: scales every element by the scalar.
//
// Defined operations:
//   &Tensor * &Tensor  -> Tensor   (matrix multiply)
//   Tensor  *= &Tensor             (matrix multiply in place)
//   &Tensor * f64      -> Tensor
//   Tensor  *= f64
//   f64     * &Tensor  -> Tensor

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|a| a * other).collect();
        Tensor {
            shape: self.shape.clone(),
            data: data.into_boxed_slice(),
        }
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, other: f64) {
        for a in self.data.iter_mut() {
            *a *= other;
        }
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        other * self
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape.len(),
            2,
            "Tensor multiplication is only for matrices"
        );
        assert_eq!(
            other.shape.len(),
            2,
            "Tensor multiplication is only for matrices"
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "The matrices should have a common dimension"
        );

        let mut new_data: Vec<f64> = vec![0.0; self.shape[0] * other.shape[1]];
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                for k in 0..self.shape[1] {
                    new_data[i * other.shape[1] + j] +=
                        self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
            }
        }

        Tensor {
            shape: vec![self.shape[0], other.shape[1]].into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }
}

impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        let result = &*self * other;
        self.shape = result.shape;
        self.data = result.data;
    }
}
