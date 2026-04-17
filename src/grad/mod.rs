pub mod core;
mod binary;
mod backward;
mod scalar;
mod unary;

pub(crate) use core::BackpropOp;

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_backward() {
        let mut x: Tensor = Tensor::linspace(1.0, 5.0, 5);
        let mut y: Tensor = Tensor::linspace(1.0, 5.0, 5);
        x.set_requires_grad(true);
        y.set_requires_grad(true);
        let z = &x * &y;
        let a = &z + &x; 
        let b = &z - &x;
        let c = &b / &a;
        c.backward();
        println!("x {:?}\n", x);
        println!("y {:?}\n", y);
        println!("z {:?}\n", z);
        println!("a {:?}\n", a);
        println!("b {:?}\n", b);
        println!("c {:?}\n", c);
    }
}
