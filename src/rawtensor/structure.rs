use std::rc::Rc;
use super::RawTensor;

// Checks whether two dimensions are compatible under broadcasting rules.
fn are_dimensions_compatible(d1: usize, d2: usize) -> bool {
    (d1 == d2) || (d1 == 1) || (d2 == 1)
}

// Computes the output shape from two broadcastable shapes. Panics if they aren't compatible.
pub(super) fn get_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Box<[usize]> {
    let len = usize::max(shape1.len(), shape2.len());
    let mut out = vec![1; len].into_boxed_slice();
    for i in 0..len {
        let d1 = if i < shape1.len() { shape1[shape1.len() - 1 - i] } else { 1 };
        let d2 = if i < shape2.len() { shape2[shape2.len() - 1 - i] } else { 1 };
        assert!(are_dimensions_compatible(d1, d2), "Shapes are not broadcastable");
        out[len - 1 - i] = usize::max(d1, d2);
    }
    out
}

pub fn contiguous_strides(shape: &[usize]) -> Box<[usize]> {
    if shape.is_empty() { return Box::from([]); }
    let mut strides: Box<[usize]> = vec![1; shape.len()].into_boxed_slice();
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub struct StridedIter<'a> {
    data: &'a [f64],
    shape: &'a [usize],
    strides: &'a [usize],
    physical: usize,
    indices: Box<[usize]>,
    done: bool,
}

impl<'a> Iterator for StridedIter<'a> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        if self.done { return None; }

        let res: f64 = self.data[self.physical];

        let mut i: usize = self.indices.len();
        while i > 0 && self.indices[i - 1] + 1 == self.shape[i - 1] {
            i -= 1;
            self.physical -= self.indices[i] * self.strides[i];
            self.indices[i] = 0;
        }
        if i == 0 {
            self.done = true;
        } else {
            i -= 1;
            self.indices[i] += 1;
            self.physical += self.strides[i];
        }

        Some(res)
    }
}

impl RawTensor {
    pub fn iter_strided(&self) -> StridedIter<'_> {
        StridedIter {
            data: &self.data,
            shape: &self.shape,
            strides: &self.strides,
            physical: 0,
            indices: vec![0; self.shape.len()].into_boxed_slice(),
            done: self.data.is_empty(),
        }
    }

    // Checks if the data are supposed to be read in the same order as they are stored in memory
    pub fn is_contiguous(&self) -> bool {
        self.strides.iter().any(|&x| x > 0) && self.strides.windows(2).all(|w| w[0] >= w[1])                                                    
    }

    // Returns a contiguous copy with elements in logical order and row-major strides.
    pub fn get_contiguous(&self) -> RawTensor {
        let new_data: Rc<[f64]> = self.iter_strided().collect();
        RawTensor::from_rc(&self.shape, &new_data)
    }

    pub fn reshape(&self, new_shape: &[usize]) -> RawTensor {
        assert!(self.is_contiguous(), "Non-contiguous tensor can't be reshaped");
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.data.len(),
            "New shape doesn't match old data length"
        );
        RawTensor::from_rc(new_shape, &self.data)
    }

    pub fn transpose(&self, perm: &[usize]) -> RawTensor {
        assert_eq!(perm.len(), self.shape.len(), "Permutation length doesn't match tensor rank");
        assert_eq!(
            { let mut v = perm.to_vec(); v.sort(); v },
            (0..perm.len()).collect::<Vec<usize>>(),
            "Permutation is not valid"
        );
        let new_shape:   Box<[usize]> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Box<[usize]> = perm.iter().map(|&i| self.strides[i]).collect();
        RawTensor {
            shape:   new_shape,
            strides: new_strides,
            data:    Rc::clone(&self.data),
        }
    }

    // Broadcasts self to new_shape by setting strides to 0 on expanded dims.
    // No data is copied; expanded dims reuse the same memory.
    pub fn expand(&self, new_shape: &[usize]) -> RawTensor {
        assert_eq!(get_broadcast_shape(&self.shape, new_shape), Box::from(new_shape), "Shapes don't match");

        let new_strides: Box<[usize]> = (0..new_shape.len())
            .rev()
            .map(|i| {
                if i >= self.shape.len() || self.shape[self.shape.len() - 1 - i] == 1 {
                    0
                } else {
                    self.strides[self.shape.len() - 1 - i]
                }
            })
            .collect();
        
        RawTensor {
            shape:   Box::from(new_shape),
            strides: new_strides,
            data:    Rc::clone(&self.data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_2d() {
        assert_eq!(&*contiguous_strides(&[3, 4]), &[4, 1]);
    }

    #[test]
    fn contiguous_strides_1d() {
        assert_eq!(&*contiguous_strides(&[5]), &[1]);
    }

    #[test]
    fn contiguous_strides_empty() {
        assert_eq!(&*contiguous_strides(&[]), &[]);
    }

    #[test]
    fn get_contigous_after_transpose() {
        let t = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = t.transpose(&[1, 0]).get_contiguous();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(&*tt.strides, &[2, 1]);
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
        // [3] expanded to [2, 3]: new leading dim is broadcast
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let e = t.expand(&[2, 3]);
        assert_eq!(e.shape(), &[2, 3]);
        assert_eq!(&*e.strides, &[0, 1]);
    }

    #[test]
    fn expand_broadcast_repeats_data() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let e = t.expand(&[2, 3]);
        let vals: Vec<f64> = e.iter_strided().collect();
        assert_eq!(vals, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn expand_incompatible_shape_panics() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        t.expand(&[2, 4]);
    }
}
