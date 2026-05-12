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
    // Returns a new RawTensor with a fresh data allocation keeping all strides structure
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
    fn approx_eq_all(a: &[f64], b: &[f64]) -> bool { 
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
    }

    #[test]
    fn basic_ops() {
        assert_eq!(*RawTensor::from_slice(&[4], &[0.0, 1.0, 4.0, 9.0]).sqrt().contiguous_data(), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(*RawTensor::from_slice(&[5], &[-3.0, -1.0, 0.0, 1.0, 3.0]).abs().contiguous_data(), [3.0, 1.0, 0.0, 1.0, 3.0]);
        assert_eq!(*RawTensor::from_slice(&[5], &[-2.0, -1.0, 0.0, 1.0, 2.0]).relu().contiguous_data(), [0.0, 0.0, 0.0, 1.0, 2.0]);
        assert_eq!(*(-&RawTensor::from_slice(&[4], &[1.0, -2.0, 0.0, 3.0])).contiguous_data(), [-1.0, 2.0, 0.0, -3.0]);

        let t = RawTensor::from_slice(&[4], &[-1.0, 0.0, 1.0, 2.0]);
        assert!(approx_eq_all(&t.exp().contiguous_data(), &[-1.0, 0.0, 1.0, 2.0].map(f64::exp)));
        assert!(approx_eq_all(&t.tanh().contiguous_data(), &[-1.0, 0.0, 1.0, 2.0].map(f64::tanh)));

        let ln_in = [0.5, 1.0, 2.0, 10.0];
        assert!(approx_eq_all(&RawTensor::from_slice(&[4], &ln_in).ln().contiguous_data(), &ln_in.map(f64::ln)));

        let sig_in = [-2.0, 0.0, 1.0, 2.0];
        assert!(approx_eq_all(&RawTensor::from_slice(&[4], &sig_in).sigmoid().contiguous_data(), &sig_in.map(|x| 1.0 / ((-x).exp() + 1.0))));
    }
   
    #[test]
    fn map_on_transposed() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let tt = t.transpose(&[1, 0]);
        let r = tt.map(|x| x * 2.0);
        assert_eq!(*r.shape, [3, 2]);
        assert_eq!(*r.strides, [1, 3]);
        assert_eq!(*r.contiguous_data(), [2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    } 

    #[test]
    fn map_on_transposed_and_expanded() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let tt = t.transpose(&[1, 0]);
        let te = tt.expand(&[3, 3, 2]);
        let r = te.map(|x| x * 2.0);
        assert_eq!(*r.shape, [3, 3, 2]);
        assert_eq!(*r.contiguous_data(), [2.0, 8.0, 4.0, 10.0, 6.0, 12.0, 2.0, 8.0, 4.0, 10.0, 6.0, 12.0, 2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    } 
}
