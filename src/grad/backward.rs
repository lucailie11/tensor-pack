use crate::Tensor;
use crate::rawtensor::RawTensor;
use std::collections::HashSet;
use std::rc::Rc;

fn topo_sort_dfs(tensor: &Tensor, sorted: &mut Vec<Tensor>, visited: &mut HashSet<usize>) {
    let id: usize = Rc::as_ptr(&tensor.0) as usize;
    if visited.contains(&id) { return; };
    visited.insert(id);

    if !tensor.tracks_grad() { return };

    for input in tensor.inputs.iter() {
        topo_sort_dfs(input, sorted, visited);
    }
    sorted.push(tensor.clone());
}

fn topo_sort(root: &Tensor) -> Vec<Tensor> {
    let mut sorted: Vec<Tensor> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    topo_sort_dfs(root, &mut sorted, &mut visited);
    sorted
}

impl Tensor {
    // Runs backpropagation from this tensor. Seeds the gradient with all-ones.
    // Only valid for scalar outputs — calling on a non-scalar tensor uses an
    // all-ones gradient, which is equivalent to summing all output elements.
    pub fn backward(&self) {
        *self.grad.borrow_mut() = Some(RawTensor::ones(self.raw.shape()));

        let topo = topo_sort(self);
        for tensor in topo.iter().rev() {
            tensor.backprop();
            if !tensor.requires_grad {
                *tensor.grad.borrow_mut() = None;
            }
        }
    }
}
