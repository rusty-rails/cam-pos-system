use super::model::Model;
use ag::optimizers;
use ag::prelude::*;
use ag::tensor_ops as T;
use ag::{ndarray::s, Context};
use autograd as ag;
use autograd::rand::prelude::SliceRandom;
use std::time::Instant;

use super::dataset::DataSet;

macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

type Tensor<'graph> = ag::Tensor<'graph, f32>;

fn conv_pool<'g>(x: Tensor<'g>, w: Tensor<'g>, b: Tensor<'g>, train: bool) -> Tensor<'g> {
    let y1 = T::conv2d(x, w, 1, 1) + b;
    let y2 = T::relu(y1);
    let y3 = T::max_pool2d(y2, 2, 0, 2);
    T::dropout(y3, 0.25, train)
}

pub fn compute_logits<'g>(
    c: &'g Context<f32>,
    input_width: isize,
    input_height: isize,
    train: bool,
) -> Tensor<'g> {
    let x = c.placeholder("x", &[-1, input_width * input_height * 3]);
    let x = x.reshape(&[-1, 3, input_width, input_height]); // 2D -> 4D
    let z1 = conv_pool(x, c.variable("w1"), c.variable("b1"), train); // map to 32 channel
    let z2 = conv_pool(z1, c.variable("w2"), c.variable("b2"), train); // map to 64 channel
    let z3 = T::reshape(z2, &[-1, 64 * input_width / 4 * input_height / 4]); // flatten
    let z4 = T::matmul(z3, c.variable("w3")) + c.variable("b3");
    T::dropout(z4, 0.25, train)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut ag::ndarray_ext::get_default_rng());
    perm
}

pub trait Trainable {
    fn train(&mut self, dataset: &DataSet, epochs: usize);
    fn evaluate(&self, dataset: &DataSet);
}

impl Trainable for Model<'_> {
    fn train(&mut self, dataset: &DataSet, epochs: usize) {
        let ((x_train, y_train), (_x_test, _y_test)) = dataset.get();
        let batch_size = 64isize;
        let num_train_samples = x_train.shape()[0];
        let num_batches = num_train_samples / batch_size as usize;

        for epoch in 0..epochs {
            let mut loss_sum = 0f32;
            timeit!({
                for i in get_permutation(num_batches) {
                    let i = i as isize * batch_size;
                    let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    let (input_width, input_height) =
                        (self.input_width as isize, self.input_height as isize);

                    self.env.run(|ctx| {
                        let logits = compute_logits(ctx, input_width, input_height, true);
                        let loss =
                            T::sparse_softmax_cross_entropy(logits, ctx.placeholder("y", &[-1, 1]));
                        let mean_loss = T::reduce_mean(loss, &[0], false);
                        let ns = ctx.default_namespace();
                        let (vars, grads) = optimizers::grad_helper(&[mean_loss], &ns);
                        let update_op = self.optimizer.get_update_op(&vars, &grads, ctx);

                        let eval_results = ctx
                            .evaluator()
                            .push(mean_loss)
                            .push(update_op)
                            .feed("x", x_batch)
                            .feed("y", y_batch)
                            .run();

                        eval_results[1].as_ref().expect("parameter updates ok");
                        loss_sum += eval_results[0].as_ref().unwrap()[0];
                    });
                }
                println!(
                    "finish epoch {}, test loss: {}",
                    epoch,
                    loss_sum / num_batches as f32
                );
            });
        }
    }

    fn evaluate(&self, dataset: &DataSet) {
        let ((_x_train, _y_train), (x_test, y_test)) = dataset.get();
        self.env.run(|ctx| {
            let logits = compute_logits(
                ctx,
                self.input_width as isize,
                self.input_height as isize,
                false,
            );
            let predictions = T::argmax(logits, -1, true);
            let accuracy = T::reduce_mean(
                &T::equal(predictions, ctx.placeholder("y", &[-1, 1])),
                &[0, 1],
                false,
            );
            println!(
                "test accuracy: {:?}",
                ctx.evaluator()
                    .push(accuracy)
                    .feed("x", x_test.view())
                    .feed("y", y_test.view())
                    .run()
            );
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LABELS: usize = 97;
    const IMAGES_PER_LABEL: usize = 21;

    #[test]
    fn test_training() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), LABELS);
        let mut model = Model::new(28, 28);
        model.train(&dataset, 10);
    }

    #[test]
    fn test_evaluate() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
        let mut model = Model::new(28, 28);
        model.train(&dataset, 10);
        model.evaluate(&dataset);
    }
}
