use crate::Tensor;
use crate::rawtensor::RawTensor;
use std::collections::HashSet;
use std::rc::Rc;

// Actual recursive topological sort function
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

// Initializations for the topological sort
fn topo_sort(root: &Tensor) -> Vec<Tensor> {
    let mut sorted: Vec<Tensor> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    topo_sort_dfs(root, &mut sorted, &mut visited);
    sorted
}

impl Tensor {
    // Runs backpropagation from this tensor
    // Starting gradient is all ones
    // Must only be called once per graph - calling it twice undefined behaviour for now
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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    fn large_graph_grad() {
        // z = (relu(x*2 - 1) + abs(y - 2)) * (x*y - 3)
        // 11 nodes: x, y, x*2, x*2-1, relu, y-2, abs, +, x*y, x*y-3, z
        // x=[1,2,3], y=[1,3,2]
        // c=relu([1,3,5])=[1,3,5], e=|[-1,1,0]|=[1,1,0], f=[2,4,5], h=[-2,3,3]
        //
        // gx = relu'(2x-1)*2*(xy-3) + (relu(2x-1)+|y-2|)*y = [-4,6,6]+[2,12,10] = [-2,18,16]
        // gy = sign(y-2)*(xy-3)     + (relu(2x-1)+|y-2|)*x = [2,3,0]+[2,8,15]   = [4,11,15]
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[1.0, 3.0, 2.0]).requires_grad();
        let a = &x * 2.0;
        let b = &a - 1.0;
        let c = b.relu();
        let d = &y - 2.0;
        let e = d.abs();
        let f = &c + &e;
        let g = &x * &y;
        let h = &g - 3.0;
        (&f * &h).backward();
        assert_eq!(grad_of(&x), [-2.0, 18.0, 16.0]);
        assert_eq!(grad_of(&y), [4.0, 11.0, 15.0]);
    }

    #[test]
    fn diamond_accumulation() {
        // z = (x*x + x*2) * (relu(x) + 1), x=[1,2,3]  — x used in 4 places
        // For x>0: z = (x²+2x)(x+1) = x³+3x²+2x  →  dz/dx = 3x²+6x+2
        // gx = [11, 26, 47]
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let a = &x * &x;
        let b = &x * 2.0;
        let c = &a + &b;
        let d = x.relu();
        let e = &d + 1.0;
        (&c * &e).backward();
        assert_eq!(grad_of(&x), [11.0, 26.0, 47.0]);
    }

    #[test]
    fn no_grad_and_intermediate() {
        // w requires_grad; x does not; z = relu(w * x) + 1
        // w=[1,2,3], x=[2,3,4] → w*x=[2,6,12] all positive → relu is identity
        // gw = x = [2,3,4]; x and all intermediate nodes have no grad after backward
        let w = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let x = Tensor::linspace(2.0, 4.0, 3);
        let y = &w * &x;
        let r = y.relu();
        let z = &r + 1.0;
        z.backward();
        assert_eq!(grad_of(&w), [2.0, 3.0, 4.0]);
        assert!(x.grad.borrow().is_none());
        assert!(y.grad.borrow().is_none());
        assert!(r.grad.borrow().is_none());
        assert!(z.grad.borrow().is_none());
    }
}


