use super::RawTensor;
use crate::utils::stride::softmax;
use std::rc::Rc;

// Axis-based operations — functions that are applied independently along one axis
// of the tensor while keeping all other axes fixed.
//
// apply_axis_inplace / apply_axis are the core primitives: they iterate over every
// (outer, inner) pair and call f on the slice of `axis_size` elements that spans
// the target axis at that position.
//
// Defined operations:
//   softmax(axis) / softmax_inplace(axis)

impl RawTensor {
    pub fn apply_axis_inplace(&mut self, axis: usize, f: impl Fn(&mut [f64], usize)) {
        assert!(axis < self.shape.len(), "axis out of bounds");

        let mut outer_size: usize = 1;
        let mut inner_size: usize = 1;
        for i in 0..self.shape.len() {
            if i < axis {
                outer_size *= self.shape[i];
            } else if i > axis {
                inner_size *= self.shape[i];
            }
        }
        let outer_size = outer_size;
        let axis_size = self.shape[axis];
        let inner_size = inner_size;


        if let Some(data) = Rc::get_mut(&mut self.data) {
            for o in 0..outer_size {
                for i in 0..inner_size {
                    f(&mut data[
                        o * axis_size * inner_size + i..
                        o * axis_size * inner_size + axis_size * inner_size], inner_size);
                }
            }
        }
    }

    pub fn apply_axis(&self, axis: usize, f: impl Fn(&mut [f64], usize)) -> RawTensor {
        let mut result = RawTensor::from_slice(&self.shape, &self.data);
        result.apply_axis_inplace(axis, f);
        result
    }

    pub fn softmax(&self, axis: usize) -> RawTensor { self.apply_axis(axis, softmax) }

    pub fn softmax_inplace(&mut self, axis: usize) { self.apply_axis_inplace(axis, softmax); }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn softmax_sums_to_one() {
        let a = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.softmax(0);
        let sum: f64 = b.data().iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn softmax_2d_rows_sum_to_one() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let b = a.softmax(1);
        let row0: f64 = b.data()[0..3].iter().sum();
        let row1: f64 = b.data()[3..6].iter().sum();
        assert!(approx_eq(row0, 1.0));
        assert!(approx_eq(row1, 1.0));
    }

    #[test]
    fn softmax_uniform_on_equal_inputs() {
        let a = RawTensor::from_slice(&[4], &[5.0, 5.0, 5.0, 5.0]);
        let b = a.softmax(0);
        assert!(b.data().iter().all(|&x| approx_eq(x, 0.25)));
    }

    #[test]
    fn softmax_large_values_stable() {
        // Without max-subtraction this would overflow to NaN/Inf
        let a = RawTensor::from_slice(&[3], &[1000.0, 1001.0, 1002.0]);
        let b = a.softmax(0);
        let sum: f64 = b.data().iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(b.data().iter().all(|&x| x.is_finite() && x > 0.0));
    }

    #[test]
    fn softmax_inplace_matches_softmax() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let expected = a.softmax(0);
        let mut b = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        b.softmax_inplace(0);
        assert!(b.data().iter().zip(expected.data()).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    #[should_panic]
    fn apply_axis_out_of_bounds() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        let _ = a.softmax(2);
    }
}
