use super::RawTensor;
use crate::utils::stride::softmax;

// Normalization operations along a single axis
// Core primitive is reduce_axis
//
// Defined operations:
//   softmax(axis)

impl RawTensor {
    //TODO
    pub fn normalize_axis(&self, axis: usize, _f: impl Fn(&mut [f64], usize, usize)) -> RawTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");
        RawTensor::randn(&[5], 0.0, 1.0)
    }

    pub fn softmax_axis(&self, axis: usize) -> RawTensor { self.normalize_axis(axis, softmax) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    #[ignore = "transform not implemented yet"]
    fn softmax_sums_to_one() {
        let a = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.softmax_axis(0);
        let sum: f64 = b.data().iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    #[ignore = "transform not implemented yet"]
    fn softmax_2d_rows_sum_to_one() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let b = a.softmax_axis(1);
        let row0: f64 = b.data()[0..3].iter().sum();
        let row1: f64 = b.data()[3..6].iter().sum();
        assert!(approx_eq(row0, 1.0));
        assert!(approx_eq(row1, 1.0));
    }

    #[test]
    #[ignore = "transform not implemented yet"]
    fn softmax_uniform_on_equal_inputs() {
        let a = RawTensor::from_slice(&[4], &[5.0, 5.0, 5.0, 5.0]);
        let b = a.softmax_axis(0);
        assert!(b.data().iter().all(|&x| approx_eq(x, 0.25)));
    }

    #[test]
    #[ignore = "transform not implemented yet"]
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
