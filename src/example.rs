use rml::Tensor;

fn main() {
    let mut x: Tensor = Tensor::linspace(1.0, 24.0, 24);
    x.reshape(&[2, 3, 4]);

    println!("{}\n{}\n", x, x.sum_axis(0));
}
