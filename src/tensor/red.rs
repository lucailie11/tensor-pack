use crate::Tensor;

// ── Private helpers ──────────────────────────────────────────────────────────
//
// These operate on plain slices so they can be reused by both the whole-tensor
// methods (sum, mean, var, std_dev) and the axis-reduction primitive
// (reduce_axis), which slices the data internally before calling them.

// Naive sequential sum. Sufficient for f64 over typical ML tensor sizes.
fn sum(data: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    for &x in data {
        sum += x;
    }
    sum
}

// Arithmetic mean: sum / n.
fn mean(data: &[f64]) -> f64 {
    sum(data) / data.len() as f64
}

// Welford's online algorithm — numerically stable single-pass mean and
// population variance. Avoids catastrophic cancellation that arises in the
// naive E[X²] - E[X]² formula when the mean is large relative to the spread.
//
// Update rule (n = number of samples seen so far, 1-indexed):
//   delta  = x - mean_prev
//   mean  += delta / n
//   M2    += delta * (x - mean_new)   ← uses the *updated* mean
//   var    = M2 / n                   ← population variance
fn mean_and_var(data: &[f64]) -> (f64, f64) {
    let mut mean: f64 = 0.0;
    let mut var: f64 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta: f64 = x - mean;
        mean += delta / (i + 1) as f64;
        var += delta * (x - mean);
    }
    (mean, var / data.len() as f64)
}

// Population variance (σ²).
fn var(data: &[f64]) -> f64 {
    mean_and_var(data).1
}

// Population standard deviation (σ).
fn std_dev(data: &[f64]) -> f64 {
    f64::sqrt(var(data))
}

// ── Tensor impl ──────────────────────────────────────────────────────────────

impl Tensor {
    // Core axis-reduction primitive. Applies `f` to every 1-D slice along
    // `axis`, producing a tensor whose shape is the input shape with `axis`
    // removed.
    //
    // Row-major indexing: the element at logical index (outer, a, inner) lives
    // at flat offset:
    //   outer * axis_size * inner_size + a * inner_size + inner
    //
    // outer_size = product of all dims before `axis`
    // inner_size = product of all dims after  `axis`
    //
    // Example: shape [3, 4, 5] reduced on axis 1 → shape [3, 5]
    fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64]) -> f64) -> Tensor {
        assert!(!self.shape.is_empty(), "reduce_axis requires at least a 1D tensor");
        assert!(axis < self.shape.len(), "axis out of bounds");

        let mut outer_size: usize = 1;
        let mut inner_size: usize = 1;
        let mut new_shape: Vec<usize> = Vec::with_capacity(self.shape.len() - 1);
        for i in 0..self.shape.len() {
            if i != axis {
                new_shape.push(self.shape[i]);
            }

            if i < axis {
                outer_size *= self.shape[i];
            } else if i > axis {
                inner_size *= self.shape[i];
            }
        }
        let new_shape = new_shape;
        let outer_size = outer_size;
        let axis_size = self.shape[axis];
        let inner_size = inner_size;

        let mut new_data: Vec<f64> = vec![0.0; outer_size * inner_size];
        for o in 0..outer_size {
            for i in 0..inner_size {
                // Gather the axis slice into a contiguous buffer before passing
                // to `f`, since the elements are strided in the flat array.
                let vec: Vec<f64> = (0..axis_size)
                    .map(|a| self.data[o * axis_size * inner_size + a * inner_size + i])
                    .collect();
                new_data[o * inner_size + i] = f(&vec)
            }
        }

        Tensor {
            shape: new_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // ── Axis reductions ──────────────────────────────────────────────────────

    // Sum of elements along `axis`. Output shape drops that dimension.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, sum)
    }

    // Mean of elements along `axis`. Output shape drops that dimension.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, mean)
    }

    // Population variance along `axis`. Output shape drops that dimension.
    pub fn var_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, var)
    }

    // Population standard deviation along `axis`. Output shape drops that dimension.
    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, std_dev)
    }

    // ── Whole-tensor reductions ──────────────────────────────────────────────

    // Sum of all elements. Returns a scalar f64.
    pub fn sum(&self) -> f64 {
        sum(&self.data)
    }

    // Arithmetic mean of all elements. Returns a scalar f64.
    pub fn mean(&self) -> f64 {
        mean(&self.data)
    }

    // Population variance (σ²) of all elements. Returns a scalar f64.
    pub fn var(&self) -> f64 {
        var(&self.data)
    }

    // Population standard deviation (σ) of all elements. Returns a scalar f64.
    pub fn std_dev(&self) -> f64 {
        std_dev(&self.data)
    }
}
