use super::RawTensor;

use std::ops::Neg;
use std::rc::Rc;

// Unary operations on Tensors
// Core primitive is map
//
// Defined operations
//   exp, ln, sqrt, abs, tanh, sigmoid, relu
//   -&RawTensor — negation

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0.0 }
}

impl RawTensor {
    pub fn map(&self, f: impl Fn(f64) -> f64) -> RawTensor {
        let new_data: Rc<[f64]> = self.data().iter().map(|x| f(*x)).collect();
        RawTensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: new_data,
        }
    }

    pub fn exp(&self) -> RawTensor     { self.map(f64::exp)  }
    pub fn ln(&self) -> RawTensor      { self.map(f64::ln)   }
    pub fn sqrt(&self) -> RawTensor    { self.map(f64::sqrt) }
    pub fn abs(&self) -> RawTensor     { self.map(f64::abs)  }
    pub fn tanh(&self) -> RawTensor    { self.map(f64::tanh) }
    pub fn sigmoid(&self) -> RawTensor { self.map(sigmoid)   }
    pub fn relu(&self) -> RawTensor    { self.map(relu)      }
}

impl Neg for &RawTensor {
    type Output = RawTensor;

    fn neg(self) -> RawTensor {
        self.map(|x| -x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    #[test]
    fn neg() {
        let a = RawTensor::from_slice(&[3], &[1.0, -2.0, 3.0]);
        let b = -&a;
        assert_eq!(b.data(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn exp_and_ln_inverses() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let b = a.exp().ln();
        assert!(b.data().iter().zip(&[1.0, 2.0, 3.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn relu_zeroes_negatives() {
        let a = RawTensor::from_slice(&[4], &[-2.0, -1.0, 0.0, 1.0]);
        let b = a.relu();
        assert_eq!(b.data(), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn sigmoid_range() {
        let a = RawTensor::from_slice(&[3], &[-100.0, 0.0, 100.0]);
        let b = a.sigmoid();
        let data = b.data();
        assert!(data[0] > 0.0 && data[0] < 0.01);
        assert!(approx_eq(data[1], 0.5));
        assert!(data[2] > 0.99 && data[2] <= 1.0);
    }

    #[test]
    fn sqrt() {
        let a = RawTensor::from_slice(&[3], &[1.0, 4.0, 9.0]);
        let b = a.sqrt();
        assert!(b.data().iter().zip(&[1.0, 2.0, 3.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn tanh_known_values() {
        let a = RawTensor::from_slice(&[3], &[0.0, 1.0, -1.0]);
        let b = a.tanh();
        assert!(approx_eq(b.data()[0], 0.0));
        assert!(approx_eq(b.data()[1], 1.0_f64.tanh()));
        assert!(approx_eq(b.data()[2], (-1.0_f64).tanh()));
    }

    #[test]
    fn abs_removes_sign() {
        let a = RawTensor::from_slice(&[4], &[-3.0, -1.0, 0.0, 2.0]);
        let b = a.abs();
        assert_eq!(b.data(), &[3.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn relu_boundary() {
        // Exactly 0.0 should pass through (>= 0 branch)
        let a = RawTensor::from_slice(&[1], &[0.0]);
        assert_eq!(a.relu().data(), &[0.0]);
    }

    #[test]
    fn neg_double_is_identity() {
        let a = RawTensor::from_slice(&[3], &[1.0, -2.0, 0.0]);
        let b = -&(-&a);
        assert_eq!(b.data(), a.data());
    }
}
