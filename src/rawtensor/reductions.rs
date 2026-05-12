use super::RawTensor;
use super::stridedops::{sum, mean, var, std_dev};

use std::rc::Rc;

// Reduction operations along a single axis
// Core primitive is reduce_axis
//
// Defined operations
// - sum
// - mean
// - var (population, divides by n)
// - std_dev (from population var)

impl RawTensor {
    // Returns a new RawTensor with a fresh data allocation, 
    // keeping the old strides structure (only removing the desired axis)
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
    pub fn var_axis(&self, axis: usize)     -> RawTensor { self.reduce_axis(axis, var) }
    pub fn std_dev_axis(&self, axis: usize) -> RawTensor { self.reduce_axis(axis, std_dev) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_axis() {
        let a = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let b = a.sum_axis(0);
        assert_eq!(*b.contiguous_data(), [5.0, 7.0, 9.0]);
        let c = a.mean_axis(0);
        assert_eq!(*c.contiguous_data(), [2.5, 3.5, 4.5]);
        let d = a.var_axis(0);
        assert_eq!(*d.contiguous_data(), [2.25; 3]);
        let e = a.std_dev_axis(0);
        assert_eq!(*e.contiguous_data(), [1.5; 3]);
    }

    #[test]
    fn reduce_axis_1() {
        let a = RawTensor::from_slice(&[1, 3], &[2.0, 4.0, 6.0]);
        let b = a.var_axis(1);
        assert_eq!(*b.shape, [1]);
        assert_eq!(*b.contiguous_data(), [8.0 / 3.0]);
    }

    #[test]
    fn reduce_axis_3d() {
        let a = RawTensor::linspace(0.0, 23.0, 24).reshape(&[2, 3, 4]);
        let b = a.sum_axis(1);
        assert_eq!(*b.shape, [2, 4]);
        assert_eq!(*b.contiguous_data(), [12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]);
    }

    #[test]
    #[should_panic]
    fn reduce_axis_out_of_bounds() {
        let _ = RawTensor::zeros(&[2, 3]).sum_axis(2);
    }

    #[test]
    fn reduce_transposed() {
        let a = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let c = a.transpose(&[1, 0]).sum_axis(0);
        assert_eq!(*c.shape, [2]);
        assert_eq!(*c.contiguous_data(), [6.0, 15.0]);
    }

    #[test]
    fn reduce_expanded() {
        let a = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let c = a.expand(&[5, 2, 3]).sum_axis(0);
        assert_eq!(*c.shape, [2, 3]);
        assert_eq!(*c.contiguous_data(), [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]);
    }

    #[test]
    fn unsqueeze_expand_reduce() {
        let a = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let b = a.unsqueeze_axis(1).expand(&[2, 5, 3]).sum_axis(1);
        assert_eq!(*b.contiguous_data(), [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]);
        let c = b.transpose(&[1, 0]).sum_axis(1);
        assert_eq!(*c.contiguous_data(), [25.0, 35.0, 45.0]);
    }

    #[test]
    fn expand_transpose_reduce() {
        let a = RawTensor::linspace(1.0, 12.0, 12).reshape(&[3, 4]);
        let b = a.unsqueeze_axis(2).expand(&[3, 4, 2]).sum_axis(2);
        assert_eq!(*b.contiguous_data(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]);
        let c = b.transpose(&[1, 0]).mean_axis(1);
        assert_eq!(*c.contiguous_data(), [10.0, 12.0, 14.0, 16.0]);
    }
}
