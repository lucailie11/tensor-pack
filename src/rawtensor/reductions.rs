use super::RawTensor;
use super::stridedops::{sum, mean, var, std_dev};

use std::rc::Rc;

// Reduction operations along a single axis
// Core primitive is reduce_axis
//
// Returns a new RawTensor with fresh new data
// If the reduced axis was actually an expanded axis (self.strides[axis] == 0)
// the new data keep their old strides
// If not, the data returned is contiguous
//
// Defined operations
// - sum
// - mean
// - var
// - std_dev

impl RawTensor {
    pub fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64], usize, usize) -> f64) -> RawTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");

        let n = self.shape[axis];
        let new_shape: Box<[usize]> = self.shape.iter().enumerate()
            .filter(|(i, _)| *i != axis).map(|(_, x)| *x).collect();

        let new_strides: Box<[usize]> = self.strides.iter().enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &x)| 
                if x > self.strides[axis] && self.strides[axis] != 0 { x / self.shape[axis] }
                else { x }
            ).collect();

        if self.strides[axis] == 0 {
            let new_data: Rc<[f64]> = self.data.iter().map(|&x| f(&[x], 0, n)).collect();

            return RawTensor {
                shape: new_shape,
                strides: new_strides,
                data: new_data,
            }
        }

        let new_data: Rc<[f64]> = self.data.iter().enumerate()
            .filter(|(i, _)| (i / self.strides[axis]).is_multiple_of(self.shape[axis]))
            .map(|(i, _)| f(&self.data[i..], self.strides[axis], n)).collect();

        RawTensor {
            shape: new_shape,
            strides: new_strides,
            data: new_data,
        }

    }

    pub fn sum_axis(&self, axis: usize)     -> RawTensor { self.reduce_axis(axis, sum) }
    pub fn mean_axis(&self, axis: usize)    -> RawTensor { self.reduce_axis(axis, mean) }
    /// Population variance (divides by n)
    pub fn var_axis(&self, axis: usize)     -> RawTensor { self.reduce_axis(axis, var) }
    /// Population standard deviation (sqrt of variance divided by n)
    pub fn std_dev_axis(&self, axis: usize) -> RawTensor { self.reduce_axis(axis, std_dev) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    #[test]
    fn sum_axis0() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.sum_axis(0);
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis1() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.sum_axis(1);
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.data(), &[6.0, 15.0]);
    } 

    #[test]
    fn mean_axis0() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.mean_axis(0);
        assert!(b.data().iter().zip(&[2.0, 3.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn var_axis0() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.var_axis(0);
        assert_eq!(b.shape(), &[2]);
        assert!(b.data().iter().zip(&[1.0, 1.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn std_dev_axis0() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.std_dev_axis(0);
        assert_eq!(b.shape(), &[2]);
        assert!(b.data().iter().zip(&[1.0, 1.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn var_axis1_known_values() {
        let a = RawTensor::from_slice(&[1, 3], &[2.0, 4.0, 6.0]);
        let b = a.var_axis(1);
        assert_eq!(b.shape(), &[1]);
        assert!(approx_eq(b.data()[0], 8.0 / 3.0));
    }

    #[test]
    fn sum_axis_3d() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let a = RawTensor::from_vec(&[2, 3, 4], data);
        let b = a.sum_axis(1);
        assert_eq!(b.shape(), &[2, 4]);
        assert_eq!(b.data()[0], 0.0 + 4.0 + 8.0);
        assert_eq!(b.data()[1], 1.0 + 5.0 + 9.0);
    }

    #[test]
    #[should_panic]
    fn reduce_axis_out_of_bounds() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        let _ = a.sum_axis(2);
    }

    #[test]
    fn reduce_transposed() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.transpose(&[1, 0]);
        let c = b.sum_axis(0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_expanded() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.expand(&[5, 2, 3]);
        let c = b.sum_axis(0);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[5.0, 10.0, 15.0, 20.0, 25.0, 30.0]);
    }

    #[test]
    fn reduce_transposed_3d() {
        let a = RawTensor::linspace(1.0, 24.0, 24);
        let b = a.reshape(&[2, 3, 4]);
        let c = b.transpose(&[1, 2, 0]);
        let d = c.sum_axis(1);
        assert_eq!(*d.contiguous_data(), [10.0, 58.0, 26.0, 74.0, 42.0, 90.0]);
    }
}
