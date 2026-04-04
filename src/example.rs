use rml::Tensor;

fn main() {
    let mut x: Tensor = Tensor::linspace(1.0, 24.0, 24);
    let mut y: Tensor = Tensor::linspace(1.0, 24.0, 24);
    x.reshape(&[4, 6]);
    y.reshape(&[6, 4]);

    println!("{}\n{}\n", x, &x.matmul(&y));
}
