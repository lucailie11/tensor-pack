use super::RawTensor;

use std::rc::Rc;
use std::ops::Index;

fn are_dimensions_compatible(d1: usize, d2: usize) -> bool {
    (d1 == d2) || (d1 == 1) || (d2 == 1)
}

// Returns the shape resulting from broadcasting shape1 and shape2. Panics if incompatible
pub(super) fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Box<[usize]> {
    let len = usize::max(shape1.len(), shape2.len());
    let mut out = vec![1; len].into_boxed_slice();
    for i in 0..len {
        let d1 = if i < shape1.len() { shape1[shape1.len() - 1 - i] } else { 1 };
        let d2 = if i < shape2.len() { shape2[shape2.len() - 1 - i] } else { 1 };
        assert!(are_dimensions_compatible(d1, d2), "shapes are not broadcastable");
        out[len - 1 - i] = usize::max(d1, d2);
    }
    out
}

// Returns the row-major strides for a contiguous tensor of the given shape
pub(super) fn strides_contiguous(shape: &[usize]) -> Box<[usize]> {
    if shape.is_empty() { return Box::from([]); }
    let mut strides: Box<[usize]> = vec![1; shape.len()].into_boxed_slice();
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub(super) fn is_stride_contiguous(strides: &[usize]) -> bool {
    strides.iter().all(|&x| x > 0) && strides.windows(2).all(|w| w[0] >= w[1])
}

impl RawTensor {
    // Returns true if the data in memory has the same order as the logical order
    pub fn is_contiguous(&self) -> bool {
        is_stride_contiguous(&self.strides)
    }

    // Returns the data in logical order. Clones the Rc if already contiguous
    pub(super) fn contiguous_data(&self) -> Rc<[f64]> {
        if self.is_contiguous() { return Rc::clone(&self.data); }
        self.iter().collect()
    }

    // Returns the strides self would have if expanded to new_shape
    pub(super) fn expand_strides(&self, new_shape: &[usize]) -> Box<[usize]> {
        assert_eq!(broadcast_shape(&self.shape, new_shape), Box::from(new_shape), "shape not broadcastable to new_shape");

        let new_strides: Box<[usize]> = (0..new_shape.len())
            .rev()
            .map(|i| {
                if i >= self.shape.len() || self.shape[self.shape.len() - 1 - i] == 1 { 0 }
                else { self.strides[self.shape.len() - 1 - i] }
            })
            .collect();

        new_strides
    }

    // Returns a new RawTensor with a new shape. Panics if tensor is not contiguous
    pub fn reshape(&self, new_shape: &[usize]) -> RawTensor {
        assert!(self.is_contiguous(), "cannot reshape a non-contiguous tensor");
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.data.len(),
            "new shape must have the same number of elements"
        );
        RawTensor::from_rc(new_shape, Rc::clone(&self.data))
    }

    // Returns a new RawTensor with dimensions permuted according to perm
    pub fn transpose(&self, perm: &[usize]) -> RawTensor {
        assert_eq!(perm.len(), self.shape.len(), "permutation length doesn't match tensor ndim");
        assert_eq!(
            { let mut v = perm.to_vec(); v.sort(); v },
            (0..perm.len()).collect::<Vec<usize>>(),
            "permutation is not valid"
        );
        RawTensor {
            shape:   perm.iter().map(|&i| self.shape[i]).collect(),
            strides: perm.iter().map(|&i| self.strides[i]).collect(),
            data:    Rc::clone(&self.data),
        }
    }

    // Expands self to new_shape. Panics if self is not broadcastable to new_shape
    pub fn expand(&self, new_shape: &[usize]) -> RawTensor {
        RawTensor { 
            shape: Box::from(new_shape), 
            strides: self.expand_strides(new_shape),
            data: Rc::clone(&self.data),
        }
    }

    // Removes a set of size-1 axes from the shape
    pub fn squeeze_axes(&self, axes: &[usize]) -> RawTensor {
        for &axis in axes {
            assert!(axis < self.shape.len(), "axis {axis} out of bounds");
            assert_eq!(self.shape[axis], 1, "cannot squeeze axis {axis} with size != 1");
        }
        let new_shape: Box<[usize]> = self.shape.iter().enumerate()
            .filter(|(i, _)| !axes.contains(i)).map(|(_, &d)| d).collect();
        let new_strides: Box<[usize]> = self.strides.iter().enumerate()
            .filter(|(i, _)| !axes.contains(i)).map(|(_, &s)| s).collect();

        RawTensor { 
            shape: new_shape,
            strides: new_strides,
            data: Rc::clone(&self.data)
        }
    }

    // Removes a single size-1 axis
    pub fn squeeze_axis(&self, axis: usize) -> RawTensor {
        self.squeeze_axes(&[axis])
    }

    // Removes all size-1 axes
    pub fn squeeze_all(&self) -> RawTensor {
        let axes: Vec<usize> = self.shape.iter().enumerate()
            .filter(|&(_, &d)| d == 1)
            .map(|(i, _)| i)
            .collect();
        self.squeeze_axes(&axes)
    }

    // Inserts a size-1 axis at the given position
    pub fn unsqueeze(&self, axis: usize) -> RawTensor {
        assert!(axis <= self.shape.len(), "axis {axis} out of bounds");
        let new_stride = if axis < self.shape.len() { self.shape[axis] * self.strides[axis] } else { 1 };
        let new_shape: Box<[usize]> = (0..=self.shape.len())
            .map(|i| if i == axis { 1 } else if i < axis { self.shape[i] } else { self.shape[i - 1] })
            .collect();
        let new_strides: Box<[usize]> = (0..=self.strides.len())
            .map(|i| if i == axis { new_stride } else if i < axis { self.strides[i] } else { self.strides[i - 1] })
            .collect();

        RawTensor { 
            shape: new_shape,
            strides: new_strides,
            data: Rc::clone(&self.data)
        }

    }

}

impl RawTensor {
    // Returns None if indices are out of bounds or wrong number of dims.
    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        if self.shape.len() != indices.len() { return None; }
        if indices.iter().zip(self.shape.iter()).any(|(i, s)| i >= s) { return None; }
        Some(self[indices])
    }
}

// Panics if indices are out of bounds or wrong number of dims.
impl Index<&[usize]> for RawTensor {
    type Output = f64;

    fn index(&self, indices: &[usize]) -> &f64 {
        assert_eq!(indices.len(), self.shape.len(), "wrong number of indices");

        let physical: usize = indices.iter().enumerate().map(|(i, &ind)| {
            assert!(ind < self.shape[i], "index out of bounds");
            ind * self.strides[i]
        }).sum();

        &self.data[physical]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_2d() {
        assert_eq!(&*strides_contiguous(&[3, 4]), &[4, 1]);
    }

    #[test]
    fn contiguous_strides_1d() {
        assert_eq!(&*strides_contiguous(&[5]), &[1]);
    }

    #[test]
    fn contiguous_strides_empty() {
        assert_eq!(&*strides_contiguous(&[]), &[]);
    }

    #[test]
    fn index_basic() {
        let t = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t[&[0usize, 0][..]], 1.0);
        assert_eq!(t[&[1usize, 2][..]], 6.0);
    }

    #[test]
    fn get_returns_none_out_of_bounds() {
        let t = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        assert!(t.get(&[5, 0]).is_none());
        assert!(t.get(&[0]).is_none());
    }

    #[test]
    fn contiguous_after_transpose_reorders_data() {
        let t = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let data = t.transpose(&[1, 0]).contiguous_data();
        assert_eq!(&*data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn contiguous_already_contiguous_no_copy() {
        let t = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        let data = t.contiguous_data();
        assert!(Rc::ptr_eq(&data, &t.data));
    }

    #[test]
    fn transpose_permutes_shape_and_strides() {
        let t = RawTensor::from_slice(&[2, 3, 4], &[0.0; 24]);
        let tt = t.transpose(&[2, 0, 1]);
        assert_eq!(tt.shape(), &[4, 2, 3]);
        assert_eq!(&*tt.strides, &[1, 12, 4]);
    }

    #[test]
    #[should_panic]
    fn transpose_invalid_perm_panics() {
        let t = RawTensor::from_slice(&[2, 3], &[0.0; 6]);
        t.transpose(&[0, 0]);
    }

    #[test]
    fn expand_sets_broadcast_strides_to_zero() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let e = t.expand(&[2, 3]);
        assert_eq!(e.shape(), &[2, 3]);
        assert_eq!(&*e.strides, &[0, 1]);
    }

    #[test]
    fn expand_broadcast_repeats_data() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let e = t.expand(&[2, 3]);
        let vals: Vec<f64> = e.iter().collect();
        assert_eq!(vals, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn expand_incompatible_shape_panics() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        t.expand(&[2, 4]);
    }

    #[test]
    fn squeeze_removes_size1_dim() {
        let t = RawTensor::from_slice(&[2, 1, 3], &[0.0; 6]);
        let s = t.squeeze_axis(1);
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    #[should_panic]
    fn squeeze_non_unit_dim_panics() {
        let t = RawTensor::from_slice(&[2, 3], &[0.0; 6]);
        t.squeeze_axis(0);
    }

    #[test]
    fn squeeze_all_removes_all_size1_dims() {
        let t = RawTensor::from_slice(&[1, 2, 1, 3, 1], &[0.0; 6]);
        assert_eq!(t.squeeze_all().shape(), &[2, 3]);
    }

    #[test]
    fn squeeze_all_no_size1_dims_unchanged() {
        let t = RawTensor::from_slice(&[2, 3], &[0.0; 6]);
        assert_eq!(t.squeeze_all().shape(), &[2, 3]);
    }

    #[test]
    fn unsqueeze_inserts_dim() {
        let t = RawTensor::from_slice(&[3, 4], &[0.0; 12]);
        assert_eq!(t.unsqueeze(0).shape(), &[1, 3, 4]);
        assert_eq!(t.unsqueeze(1).shape(), &[3, 1, 4]);
        assert_eq!(t.unsqueeze(2).shape(), &[3, 4, 1]);
    }

    #[test]
    fn unsqueeze_preserves_contiguity() {
        let t = RawTensor::from_slice(&[3, 4], &[0.0; 12]);
        assert!(t.unsqueeze(1).is_contiguous());
    }

    #[test]
    #[should_panic]
    fn unsqueeze_out_of_bounds_panics() {
        let t = RawTensor::from_slice(&[3, 4], &[0.0; 12]);
        t.unsqueeze(3);
    }
}
