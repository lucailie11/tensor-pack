use rml::Tensor;

fn main() {
    let mut x: Tensor = Tensor::linspace(1.0, 6.0, 6);
    let mut y: Tensor = Tensor::linspace(7.0, 12.0, 6);

    x += &y;
    y += &x;
    println!("{}\n{}\n{}\n", x, y, &x - &y);
}
