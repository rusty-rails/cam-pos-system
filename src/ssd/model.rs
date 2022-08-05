// source: https://github.com/raskr/rust-autograd/blob/master/examples/cnn_mnist.rs
use ag::ndarray_ext as array;
use ag::optimizers;
use ag::prelude::*;
use autograd as ag;

pub struct Model<'a> {
    pub env: ag::VariableEnvironment<'a, f32>,
    pub optimizer: optimizers::Adam<f32>,
    pub input_width: usize,
    pub input_height: usize,
}

impl Model<'_> {
    pub fn new(input_width: usize, input_height: usize) -> Model<'static> {
        let mut env = ag::VariableEnvironment::<f32>::new();
        let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
        env.name("w1")
            .set(rng.random_normal(&[32, 3, 3, 3], 0., 0.1));
        env.name("w2")
            .set(rng.random_normal(&[64, 32, 3, 3], 0., 0.1));
        env.name("w3")
            .set(rng.glorot_uniform(&[64 * input_width / 4 * input_height / 4, 10]));
        env.name("b1")
            .set(array::zeros(&[1, 32, input_width, input_height]));
        env.name("b2")
            .set(array::zeros(&[1, 64, input_width / 2, input_height / 2]));
        env.name("b3").set(array::zeros(&[1, 10]));

        let adam = optimizers::Adam::default(
            "my_adam",
            env.default_namespace().current_var_ids(),
            &mut env,
        );

        Model {
            env,
            optimizer: adam,
            input_width,
            input_height,
        }
    }
}
