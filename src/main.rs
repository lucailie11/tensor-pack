mod tensor;
use tensor::Tensor;

fn main() {
    let a: Tensor = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b: Tensor = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    let mut c = &b + &a;
    c += &a;
    println!("{:?}{:?}{:?}", a.data, b.data, c.data);
}
