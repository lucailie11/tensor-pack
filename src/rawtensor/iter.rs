use super::RawTensor;

// Iterates elements in logical order in O(self.len()).
// Supports non-contiguous tensors (transposed, expanded) via strides.

pub struct LogicalIndices {
    shape: Box<[usize]>,
    strides: Box<[usize]>,
    indices: Box<[usize]>,
    flat: usize,
    done: bool,
}

impl Iterator for LogicalIndices {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.done { return None; }

        let res: usize = self.flat;

        let mut i: usize = self.indices.len();
        while i > 0 && self.indices[i - 1] + 1 == self.shape[i - 1] {
            i -= 1;
            self.flat -= self.indices[i] * self.strides[i];
            self.indices[i] = 0;
        }
        if i == 0 {
            self.done = true;
        } else {
            i -= 1;
            self.indices[i] += 1;
            self.flat += self.strides[i];
        }

        Some(res)
    }
}

impl LogicalIndices {
    pub fn new(shape: Box<[usize]>, strides: Box<[usize]>) -> LogicalIndices {
        LogicalIndices {
            done: shape.contains(&0),
            indices: vec![0; shape.len()].into_boxed_slice(),
            shape,
            strides,
            flat: 0,
        }
    }
}

impl RawTensor {
    fn logical_indices(&self) -> LogicalIndices {
        LogicalIndices::new(self.shape.clone(), self.strides.clone())
    }

    pub fn iter(&self) -> impl Iterator<Item = f64> {
        self.logical_indices().map(|p| self.data[p])
    }
                                                                                        
    pub fn iter_indexed(&self) -> impl Iterator<Item = (usize, f64)> {
        self.logical_indices().map(|p| (p, self.data[p]))
    }
}
