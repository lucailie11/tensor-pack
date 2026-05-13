use super::RawTensor;

use std::ops::Index;
use std::rc::Rc;

// --- Helper Functions --- 

// Broadcastable dimensions according to broadcasting rules
fn are_dimensions_broadcastable(d1: usize, d2: usize) -> bool {
    (d1 == d2) || (d1 == 1) || (d2 == 1)
}

// Returns the shape resulting from broadcasting shape1 and shape2. Panics if incompatible
pub(super) fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Box<[usize]>> {
    let len = usize::max(shape1.len(), shape2.len());
    let mut out = vec![1; len].into_boxed_slice();
    for i in 0..len {
        let d1 = if i < shape1.len() { shape1[shape1.len() - 1 - i] } else { 1 };
        let d2 = if i < shape2.len() { shape2[shape2.len() - 1 - i] } else { 1 };
        if !are_dimensions_broadcastable(d1, d2) { return None }
        out[len - 1 - i] = usize::max(d1, d2);
    }
    Some(out)
}

// Returns the row-major strides for a contiguous tensor of the given shape
pub(super) fn strides_contiguous(shape: &[usize]) -> Box<[usize]> {
    if shape.is_empty() { return Box::from([]); }
    let mut strides: Box<[usize]> = vec![1; shape.len()].into_boxed_slice();
    for i in (0..shape.len() - 1).rev() { strides[i] = strides[i + 1] * shape[i + 1]; }
    strides
}

// Returns the strides t will have after being expanded to new_shape
pub(super) fn expanded_strides(t: &RawTensor, new_shape: &[usize]) -> Option<Box<[usize]>> {
    let broadcast_shape: Option<Box<[usize]>> = broadcast_shape(&t.shape, new_shape);
    match broadcast_shape {
        None => return None,
        Some(s) => { if *s != *new_shape { return None } },
    }

    let new_strides: Box<[usize]> = (0..new_shape.len()).rev()
        .map(|i| {
            if i >= t.shape.len() || t.shape[t.shape.len() - 1 - i] == 1 
            { 0 } else { t.strides[t.shape.len() - 1 - i] }
        })
        .collect();

    Some(new_strides)
}

pub(super) fn is_data_contiguous(strides: &[usize]) -> bool {
    strides.iter().all(|&x| x > 0) && strides.windows(2).all(|w| w[0] >= w[1])
}

impl RawTensor {
    // Returns true if the data in memory has the same order as the logical order
    pub fn is_contiguous(&self) -> bool {
        is_data_contiguous(&self.strides)
    }

    // Returns the data in logical order. Clones the Rc if already contiguous
    pub fn contiguous_data(&self) -> Rc<[f64]> {
        if self.is_contiguous() { return Rc::clone(&self.data); }
        self.iter().collect()
    }

    // Returns a new RawTensor with data in logical order
    pub fn contiguous(&self) -> RawTensor {
        RawTensor::from_rc(&self.shape, self.contiguous_data())
    }

    // Returns a new RawTensor with a new shape. Panics if tensor is not contiguous
    pub fn reshape(&self, new_shape: &[usize]) -> RawTensor {
        assert!(self.is_contiguous(), "cannot reshape a non-contiguous tensor. call .contiguous() first");
        assert_eq!(new_shape.iter().product::<usize>(), self.data.len(), "new shape must have the same number of elements");
        RawTensor::from_rc(new_shape, Rc::clone(&self.data))
    }

    // Returns a new RawTensor with dimensions permuted (new dim_i comes from old dim_perm[i])
    pub fn transpose(&self, perm: &[usize]) -> RawTensor {
        assert_eq!(perm.len(), self.ndim(), "permutation length doesn't match tensor ndim");
        assert_eq!( { let mut v = perm.to_vec(); v.sort(); v },
            (0..perm.len()).collect::<Vec<usize>>(),
            "permutation is not valid"
        );
        RawTensor {
            shape: perm.iter().map(|&i| self.shape[i]).collect(),
            strides: perm.iter().map(|&i| self.strides[i]).collect(),
            data: Rc::clone(&self.data),
        }
    }

    // Expands self to new_shape. Panics if self is not broadcastable to new_shape
    pub fn expand(&self, new_shape: &[usize]) -> RawTensor {
        RawTensor {
            shape: Box::from(new_shape),
            strides: expanded_strides(self, new_shape).expect("old shape not broadcastable into new shape"),
            data: Rc::clone(&self.data),
        }
    }
    //
    // Removes a single size-1 axis
    pub fn squeeze(&self, axis: usize) -> RawTensor {
        assert!(axis < self.shape.len(), "axis {axis} out of bounds");
        assert_eq!(self.shape[axis], 1, "cannot squeeze axis {axis} with size != 1");

        let new_shape: Box<[usize]> = self.shape.iter().enumerate()
            .filter(|(i, _)| axis != *i).map(|(_, &d)| d).collect();
        let new_strides: Box<[usize]> = self .strides.iter().enumerate()
            .filter(|(i, _)| axis != *i).map(|(_, &s)| s).collect();

        RawTensor {
            shape: new_shape,
            strides: new_strides,
            data: Rc::clone(&self.data),
        }
    }

    // Inserts a size-1 axis at the given position
    pub fn unsqueeze(&self, axis: usize) -> RawTensor {
        assert!(axis <= self.shape.len(), "axis {axis} out of bounds");

        let new_stride = if axis < self.shape.len() 
            { self.shape[axis] * self.strides[axis] } else { 1 };

        let new_shape: Box<[usize]> = (0..=self.shape.len())
            .map(|i| {
                if i == axis { 1 }
                else if i < axis { self.shape[i] } 
                else { self.shape[i - 1] }
            })
            .collect();
        let new_strides: Box<[usize]> = (0..=self.strides.len())
            .map(|i| {
                if i == axis { new_stride } 
                else if i < axis { self.strides[i] } 
                else { self.strides[i - 1] }
            })
            .collect();

        RawTensor {
            shape: new_shape,
            strides: new_strides,
            data: Rc::clone(&self.data),
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
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");

        let physical: usize = indices .iter() .enumerate()
            .map(|(i, &ind)| {
                assert!(ind < self.shape[i], "index out of bounds");
                ind * self.strides[i]
            })
            .sum();

        &self.data[physical]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper Functions --- 

    #[test]
    fn are_dimensions_broadcastable_test() {
        assert!(are_dimensions_broadcastable(1, 4));
        assert!(are_dimensions_broadcastable(5, 1));
        assert!(are_dimensions_broadcastable(1, 1));
        assert!(are_dimensions_broadcastable(3, 3));
        assert!(!are_dimensions_broadcastable(2, 3));
    }

    #[test]
    fn strides_contiguous_test() {
        assert_eq!(*strides_contiguous(&[]),        []);
        assert_eq!(*strides_contiguous(&[5]),       [1]);
        assert_eq!(*strides_contiguous(&[3, 4]),    [4, 1]);
        assert_eq!(*strides_contiguous(&[2, 3, 4]), [12, 4, 1]);
    }

    #[test]
    fn broadcast_rules() {
        assert_eq!(*broadcast_shape(   &[5, 4],    &[5, 4]).unwrap(),    [5, 4]);
        assert_eq!(*broadcast_shape(&[1, 2, 3], &[5, 2, 1]).unwrap(), [5, 2, 3]);
        assert_eq!(*broadcast_shape(   &[2, 3], &[5, 2, 1]).unwrap(), [5, 2, 3]);
        assert_eq!(*broadcast_shape(&[5, 2, 3], &[5, 2, 1]).unwrap(), [5, 2, 3]);
        assert!(broadcast_shape(   &[5, 4], &[5, 3]).is_none());
        assert!(broadcast_shape(   &[2, 4], &[2, 2]).is_none());
        assert!(broadcast_shape(&[1, 2, 3], &[5, 2]).is_none());
    }

    #[test]
    fn is_contiguous_rules() {
        assert!(RawTensor::zeros(&[2, 3]).is_contiguous());
        assert!(!RawTensor::zeros(&[2, 3]).transpose(&[1, 0]).is_contiguous());
        assert!(RawTensor::zeros(&[2, 3]).transpose(&[1, 0]).transpose(&[1, 0]).is_contiguous());
        assert!(!RawTensor::zeros(&[2, 3]).expand(&[5, 2, 3]).is_contiguous());
        assert!(RawTensor::zeros(&[2, 3]).unsqueeze(2).is_contiguous());
        assert!(RawTensor::zeros(&[2, 3]).unsqueeze(2).squeeze(2).is_contiguous());
    }

    #[test]
    fn expanded_strides_test() {
        let t = RawTensor::zeros(&[3]);
        assert_eq!(*expanded_strides(&t, &[2, 3]).unwrap(), [0, 1]);
        let t = RawTensor::zeros(&[1, 4]);
        assert_eq!(*expanded_strides(&t, &[3, 2, 4]).unwrap(), [0, 0, 1]);
        let t = RawTensor::zeros(&[3]);
        assert!(expanded_strides(&t, &[3, 2]).is_none());
        let t = RawTensor::zeros(&[1, 2, 2]);
        assert!(expanded_strides(&t, &[3, 2, 4]).is_none());
    }

    #[test]
    fn correct_contiguous_data() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        assert_eq!(*t.contiguous_data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t = t.transpose(&[1, 0]);
        assert_eq!(*t.contiguous_data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let t = t.unsqueeze(2);
        assert_eq!(*t.contiguous_data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let t = t.squeeze(2);
        assert_eq!(*t.contiguous_data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let t = t.expand(&[3, 3, 2]);
        assert_eq!(*t.contiguous_data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
    
    #[test]
    fn correct_contiguous_data_rc() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let s = t.transpose(&[1, 0]);
        assert_ne!(Rc::as_ptr(&t.contiguous_data()), Rc::as_ptr(&s.contiguous_data()));
        let u = s.transpose(&[1, 0]);
        assert_eq!(Rc::as_ptr(&t.contiguous_data()), Rc::as_ptr(&u.contiguous_data()));
    }

    // --- Methods ---

    #[test]
    fn reshape_basic() {
        let t = RawTensor::linspace(1.0, 6.0, 6);
        let r = t.reshape(&[2, 3]);
        assert_eq!(*r.shape, [2, 3]);
        assert_eq!(*r.strides, [3, 1]);
        assert_eq!(*r.contiguous_data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_0d_to_1d() {
        let t = RawTensor::from_scalar(5.0);
        let r = t.reshape(&[1]);
        assert_eq!(*r.shape, [1]);
        assert_eq!(*r.contiguous_data(), [5.0]);
    }

    #[test]
    fn reshape_on_double_transposed() {
        let t = RawTensor::zeros(&[2, 3]).transpose(&[1, 0]).transpose(&[1, 0]);
        let s = t.reshape(&[6]);
        assert_eq!(*s.shape, [6]);
        assert_eq!(*s.strides, [1]);
    }

    #[test]
    #[should_panic]
    fn reshape_on_transposed_panics() {
        let t = RawTensor::zeros(&[2, 3]).transpose(&[1, 0]);
        t.reshape(&[6]);
    }

    #[test]
    #[should_panic]
    fn reshape_on_expand_panics() {
        let t = RawTensor::zeros(&[2, 3]).expand(&[5, 2, 3]);
        t.reshape(&[6]);
    }

    #[test]
    #[should_panic]
    fn reshape_element_count_mismatch_panics() {
        let t = RawTensor::zeros(&[6]);
        t.reshape(&[2, 4]);
    }

    #[test]
    fn transpose_shape_and_strides() {
        let t = RawTensor::zeros(&[2, 3, 4]);
        let tt = t.transpose(&[2, 0, 1]);
        assert_eq!(*tt.shape, [4, 2, 3]);
        assert_eq!(*tt.strides, [1, 12, 4]);
    }

    #[test]
    fn transpose_logical_values() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let tt = t.transpose(&[1, 0]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t[&[i, j]], tt[&[j, i]]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn transpose_duplicate_index_panics() {
        let t = RawTensor::from_slice(&[2, 3], &[0.0; 6]);
        t.transpose(&[0, 0]);
    }

    #[test]
    #[should_panic]
    fn transpose_wrong_length_panics() {
        let t = RawTensor::from_slice(&[2, 3], &[0.0; 6]);
        t.transpose(&[0]);
    }

    #[test]
    #[should_panic]
    fn transpose_not_perm_panics() {
        let t = RawTensor::from_slice(&[2, 3], &[0.0; 6]);
        t.transpose(&[1, 2]);
    }

    #[test]
    fn expand_broadcast_strides_and_shape() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let e = t.expand(&[2, 3]);
        assert_eq!(*e.shape, [2, 3]);
        assert_eq!(*e.strides, [0, 1]);
    }

    #[test]
    fn expand_logical_values_repeated() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        assert_eq!(*t.expand(&[2, 3]).contiguous_data(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn expand_0d_scalar_to_1d() {
        let t = RawTensor::from_scalar(5.0);
        assert_eq!(*t.expand(&[3]).contiguous_data(), [5.0, 5.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn expand_incompatible_shape_panics() {
        RawTensor::zeros(&[3]).expand(&[2, 4]);
    }

    #[test]
    fn squeeze_axis_removes_dim_and_strides() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 1, 3]);
        let s = t.squeeze(1);
        assert_eq!(*s.shape, [2, 3]);
        assert_eq!(*s.strides, [3, 1]);
    }

    #[test]
    fn squeeze_axis_preserves_logical_values() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 1, 3]);
        let s = t.squeeze(1);
        assert_eq!(*s.shape, [2, 3]);
        assert_eq!(*s.strides, [3, 1]);
        assert_eq!(*s.contiguous_data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }


    #[test]
    #[should_panic]
    fn squeeze_non_unit_dim_panics() {
        RawTensor::zeros(&[2, 3]).squeeze(0);
    }

    #[test]
    #[should_panic]
    fn squeeze_out_of_bounds_panics() {
        RawTensor::zeros(&[2, 3]).squeeze(2);
    }

    #[test]
    fn unsqueeze_inserts_at_front_mid_end() {
        let t = RawTensor::zeros(&[3, 4]);
        assert_eq!(*t.unsqueeze(0).shape, [1, 3, 4]);
        assert_eq!(*t.unsqueeze(1).shape, [3, 1, 4]);
        assert_eq!(*t.unsqueeze(2).shape, [3, 4, 1]);
    }

    #[test]
    fn unsqueeze_preserves_logical_values() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let s = t.unsqueeze(1);
        assert_eq!(*s.contiguous_data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn unsqueeze_out_of_bounds_panics() {
        RawTensor::zeros(&[3, 4]).unsqueeze(3);
    }

    #[test]
    fn index_basic() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t[&[i, j]], (1 + i * 3 + j) as f64);
            }
        }
    }

    #[test]
    fn index_on_transposed() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let tt = t.transpose(&[1, 0]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tt[&[j, i]], (1 + i * 3 + j) as f64);
            }
        }
    }

    #[test]
    fn index_on_expand() {
        let t = RawTensor::linspace(1.0, 3.0, 3);
        let te = t.expand(&[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(te[&[i, j]], (1 + j) as f64);
            }
        }
    }

    #[test]
    #[should_panic]
    fn index_out_of_bounds_panics() {
        let t = RawTensor::zeros(&[2, 3]);
        let _ = t[&[2, 0]];
    }

    #[test]
    #[should_panic]
    fn index_wrong_ndim() {
        let t = RawTensor::zeros(&[2, 3]);
        let _ = t[&[0]];
    }

    #[test]
    fn get_valid_index() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        assert_eq!(t.get(&[1, 2]), Some(6.0));
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let t = RawTensor::zeros(&[2, 3]);
        assert!(t.get(&[5, 0]).is_none());
    }

    #[test]
    fn get_wrong_ndim_returns_none() {
        let t = RawTensor::zeros(&[2, 3]);
        assert!(t.get(&[0]).is_none());
    }

    #[test]
    fn get_0d_tensor() {
        let t = RawTensor::from_scalar(42.0);
        assert_eq!(t.get(&[]), Some(42.0));
    }

    #[test]
    fn contiguous_on_transposed() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).transpose(&[1, 0]);
        assert!(!t.is_contiguous());
        let c = t.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(*c.shape, [3, 2]);
        assert_eq!(*c.strides, [2, 1]);
        assert_eq!(*c.contiguous_data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn contiguous_on_expanded() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]).expand(&[4, 3]);
        assert!(!t.is_contiguous());
        let c = t.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(*c.shape, [4, 3]);
        assert_eq!(*c.strides, [3, 1]);
        assert_eq!(*c.contiguous_data(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
}
