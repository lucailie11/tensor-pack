use rml::Tensor;

fn main() {
    let mut x: Tensor = Tensor::linspace(1.0, 9.0, 9);
    let mut y: Tensor = Tensor::linspace(7.0, 16.0, 9);

    x.reshape(&[3, 3]);
    y = x.mean_axis(1);
    y.reshape(&[3, 1]);
    println!("{}\n{}\n{}", x, y, &x + &y);

    let a: Tensor = Tensor::randn(&[3000], 0.0, 10.0);
    println!("{}\n", a);
    println!("{}", a.std_dev_axis(0))
}
