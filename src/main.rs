mod tensor;
use tensor::Tensor;

fn main() {
    let a: Tensor = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b: Tensor = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    let d: Tensor = Tensor::linspace(15.0, 20.0, 4);
    let mut c = &b + &a;
    c += &a;
    println!("{:?} {:?}{:?}{:?}", a.data, b.data, c.data, d.data);
}
