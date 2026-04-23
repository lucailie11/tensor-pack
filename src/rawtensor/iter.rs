use super::RawTensor;

// Iterates elements in logical order using an odometer-style index counter.
// Supports non-contiguous tensors (transposed, expanded) via strides.
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
    pub fn iter(&self) -> StridedIter<'_> {
        StridedIter {
            data: &self.data,
            shape: &self.shape,
            strides: &self.strides,
            physical: 0,
            indices: vec![0; self.shape.len()].into_boxed_slice(),
            done: self.data.is_empty(),
        }
    }
}
