use super::RawTensor;
use std::rc::Rc;

// Normalization operations along a single axis
// Core primitive is reduce_axis
//
// Defined operations:
//   - softmax(axis)

pub fn softmax(old_data: &[f64], new_data: &mut [f64], step: usize, n: usize) {
    if n == 0 { return; }

    if step == 0 { 
        new_data[0] = 1.0 / n as f64;
        return;
    }

    let max: f64 = old_data.iter().step_by(step).take(n).copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = old_data.iter().step_by(step).take(n).map(|&x| f64::exp(x - max)).collect();
    let sum: f64 = exps.iter().sum();
    new_data.iter_mut().step_by(step).take(n).zip(exps).for_each(|(x, e)| *x = e / sum);
}

impl RawTensor {
    // Returns a new RawTensor with a fresh data allocation keeping all strides structure
    pub(super) fn normalize_axis(
        &self,
        axis: usize,
        f: impl Fn(&[f64], &mut [f64], usize, usize),
    ) -> RawTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");

        let n = self.shape[axis];
        if self.strides[axis] == 0 {
            let mut new_data: Box<[f64]> = vec![0.0; self.data.len()].into_boxed_slice();

            self.data.iter().enumerate()
                .for_each(|(i, _)| f(&self.data[i..], &mut new_data[i..], 0, n));

            return RawTensor {
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                data: Rc::from(new_data),
            };
        }

        let mut new_data: Box<[f64]> = vec![0.0; self.data.len()].into_boxed_slice();

        self.data.iter().enumerate()
            .filter(|(i, _)| (i % (self.shape[axis] * self.strides[axis])) < self.strides[axis])
            .for_each(|(i, _)| f(&self.data[i..], &mut new_data[i..], self.strides[axis], n));

        RawTensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: Rc::from(new_data),
        }
    }

    pub fn softmax_axis(&self, axis: usize) -> RawTensor {
        self.normalize_axis(axis, softmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_all(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn softmax_uniform() {
        assert_eq!(
            *RawTensor::full(&[4], 5.0).softmax_axis(0).contiguous_data(),
            [1.0 / 4.0; 4]
        );
        assert_eq!(
            *RawTensor::zeros(&[2, 3]).softmax_axis(1).contiguous_data(),
            [1.0 / 3.0; 6]
        );
    }

    #[test]
    fn softmax_lns() {
        let input = [0.0f64, 2.0f64.ln(), 3.0f64.ln()];
        let s = RawTensor::from_slice(&[3], &input).softmax_axis(0);
        assert_eq!(*s.contiguous_data(), [1.0 / 6.0, 1.0 / 3.0, 0.5]);
    }

    #[test]
    fn softmax_values() {
        let input = [-1.0, -4.0, 0.0, 1.0, -2.0];
        let s = RawTensor::from_slice(&[5], &input).softmax_axis(0);
        assert!(approx_eq_all(
            &s.contiguous_data(),
            &[0.086768, 0.00432, 0.23586, 0.641133, 0.03192]
        ));
    }

    #[test]
    fn softmax_big_values() {
        let b = RawTensor::from_slice(&[3], &[1000.0, 1001.0, 1002.0]).softmax_axis(0);
        let data = b.contiguous_data();
        assert!((data.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(data.iter().all(|&x| x.is_finite() && x > 0.0));
    }
    #[test]
    fn softmax_small_values() {
        let s = RawTensor::from_slice(&[3], &[-1000.0, -1001.0, -1002.0]).softmax_axis(0);
        let data = s.contiguous_data();
        assert!((data.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(
            data.iter()
                .all(|&x| x.is_finite() && (0.0..1.0).contains(&x))
        );
    }

    #[test]
    fn softmax_transposed() {
        let a = RawTensor::linspace(0.0, 5.0, 6).reshape(&[2, 3]);
        let t = a.transpose(&[1, 0]);
        let b = t.softmax_axis(1);
        assert_eq!(*b.shape, [3, 2]);
        assert_eq!(*b.strides, [1, 3]);
        let e3 = 3.0f64.exp();
        let lo = 1.0 / (1.0 + e3);
        let hi = e3 / (1.0 + e3);
        assert!(approx_eq_all(
            &b.contiguous_data(),
            &[lo, hi, lo, hi, lo, hi]
        ));
    }

    #[test]
    fn softmax_expanded_axis() {
        let a = RawTensor::linspace(1.0, 3.0, 3)
            .reshape(&[1, 3])
            .expand(&[4, 3]);
        let s = a.softmax_axis(0);
        assert_eq!(*s.shape, [4, 3]);
        assert_eq!(*s.contiguous_data(), [0.25; 12]);
    }

    #[test]
    fn softmax_transpose_expand() {
        let a = RawTensor::from_slice(&[2, 1], &[0.0, 2.0f64.ln()]);
        let e = a.transpose(&[1, 0]).expand(&[3, 2]);
        let s = e.softmax_axis(1);
        let t = &s * 3.0;
        assert_eq!(*t.shape, [3, 2]);
        assert_eq!(*t.contiguous_data(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn softmax_out_of_bounds() {
        RawTensor::zeros(&[2, 3]).softmax_axis(2);
    }
}
