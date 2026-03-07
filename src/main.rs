mod tensor;
use tensor::Tensor;

fn main() {
    let a: Tensor = Tensor::randn(&[2, 2], 0.0, 10.0);
    let c: Tensor = Tensor::randn(&[2, 2], 0.0, 10.0);
    let b: Tensor = Tensor::randn(&[2, 2], 0.0, 10.0);
    let d: Tensor = Tensor::randn(&[2, 2], 0.0, 10.0);
    println!("{:?}\n{:?}\n{:?}\n{:?}\n", a.data, b.data, c.data, d.data);

    let e = &a - &b;
    let mut f = -&a;
    f.reshape(&[4]);
    let mut g = &f / 0.2;
    g.exp_inplace();
    let mut h = &g / 1.0;
    h /= 2.0;

    println!("{:?}\n{:?}\n{:?}\n{:?}\n", e, f, g, h);

    let mut x: Tensor = Tensor::randn(&[2, 2], 0.0, 10.0);
    let y: Tensor = Tensor::randn(&[2, 2], 0.0, 10.0);
    let z = &x * &y;
    println!("{}\n {}\n {}\n", x, y, z);
    x *= &y;
    println!("{}\n", x);
}
