use std::rc::Rc;

use super::RawTensor;
use super::stridedops::softmax;

// Normalization operations along a single axis
// Core primitive is reduce_axis
// Returns a new RawTensor with fresh new data which keeps its strides
//
// Defined operations:
//   softmax(axis)

impl RawTensor {
    pub(super) fn normalize_axis(&self, axis: usize, f: impl Fn(&[f64], &mut [f64], usize, usize)) -> RawTensor {
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
            }
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

    pub fn softmax_axis(&self, axis: usize) -> RawTensor { self.normalize_axis(axis, softmax) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn softmax_sums_to_one() {
        let a = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.softmax_axis(0);
        let sum: f64 = b.data().iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn softmax_2d_rows_sum_to_one() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let b = a.softmax_axis(1);
        let row0: f64 = b.data()[0..3].iter().sum();
        let row1: f64 = b.data()[3..6].iter().sum();
        assert!(approx_eq(row0, 1.0));
        assert!(approx_eq(row1, 1.0));
    }

    #[test]
    fn softmax_uniform_on_equal_inputs() {
        let a = RawTensor::from_slice(&[4], &[5.0, 5.0, 5.0, 5.0]);
        let b = a.softmax_axis(0);
        assert!(b.data().iter().all(|&x| approx_eq(x, 0.25)));
    }

    #[test]
    fn softmax_large_values_stable() {
        let a = RawTensor::from_slice(&[3], &[1000.0, 1001.0, 1002.0]);
        let b = a.softmax_axis(0);
        let sum: f64 = b.data().iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(b.data().iter().all(|&x| x.is_finite() && x > 0.0));
    }

    #[test]
    #[should_panic]
    fn apply_axis_out_of_bounds() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        let _ = a.softmax_axis(2);
    }
}
