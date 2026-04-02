use crate::Tensor;

fn softmax(data: &mut [f64], step: usize) {
    let max: f64 = data
        .iter()
        .step_by(step)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    for x in data.iter_mut().step_by(step) {
        *x -= max;
    }

    let sum: f64 = data
        .iter()
        .step_by(step)
        .map(|x| f64::exp(*x))
        .sum();

    for x in data.iter_mut().step_by(step) {
        *x = f64::exp(*x) / sum;
    }
}

impl Tensor {
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

        for o in 0..outer_size {
            for i in 0..inner_size {
                f(&mut self.data[o * axis_size * inner_size + i..], inner_size);
            }
        } 
    }

    pub fn apply_axis(&self, axis: usize, f: impl Fn(&mut [f64], usize)) -> Tensor {
        assert!(axis < self.shape.len(), "axis out of bounds");
        let mut result = Tensor::new(&self.shape, &self.data);
        result.apply_axis_inplace(axis, f);
        result
    }

    pub fn softmax_inplace(&mut self, axis: usize) {
        self.apply_axis_inplace(axis, softmax);
    }
 
    pub fn softmax(&self, axis: usize) -> Tensor {
        self.apply_axis(axis, softmax)
    }
}
