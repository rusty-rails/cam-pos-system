// source: https://github.com/raskr/rust-autograd/blob/master/examples/cnn_mnist.rs
use super::predictable::detect_objects;
use crate::detection::Detection;
use crate::detector::Detector;
use crate::ssd::predictable::Predictable;
use ag::ndarray_ext as array;
use ag::optimizers;
use ag::prelude::*;
use autograd as ag;
use image::DynamicImage;

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

    pub fn save(&self, filename: &str) {
        self.env.save(filename).unwrap();
    }

    pub fn load(&mut self, filename: &str) {
        match ag::VariableEnvironment::<f32>::load(&filename) {
            Ok(loaded_env) => self.env = loaded_env,
            Err(e) => println!("{}", e),
        };
    }
}

impl Detector for Model<'_> {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection> {
        let image = image.to_rgb8();
        let window_size = self.input_width as u32;
        let cols = 2 * (image.width() / window_size);
        let rows = 2 * (image.height() / window_size);
        let predictions = self.predict_image(image);
        detect_objects(cols, rows, predictions, window_size)
    }
}

#[cfg(test)]
mod tests {
    use super::super::window_crop;
    use super::*;
    use crate::ssd::dataset::DataSet;
    use crate::ssd::trainable::Trainable;
    use image::open;
    use std::path::Path;

    #[test]
    fn test_load_and_save() {
        let filename = "out/model.json";
        let mut model = Model::new(32, 32);
        model.load(filename);
        model.save(filename);
        assert!(Path::new(filename).exists());
    }

    #[ignore = "long train time"]
    #[test]
    fn test_detector_via_hard_negative_samples() {
        let model_filename = "out/model.json";
        let window_size = 32;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (280, 370));
        let marker1 = window_crop(&webcam1, window_size, window_size, (540, 90));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 25));

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(true);
        let mut model = Model::new(32, 32);
        model.load(model_filename);
        model.train(&dataset, 100);
        dataset.dataset.generate_hard_negative_samples(&model, 1);
        dataset.dataset.generate_hard_negative_samples(&model, 2);
        dataset.dataset.generate_hard_negative_samples(&model, 3);
        dataset.dataset.generate_hard_negative_samples(&model, 5);
        model.train(&dataset, 50);
        dataset.dataset.generate_hard_negative_samples(&model, 5);
        model.train(&dataset, 50);
        dataset.dataset.generate_hard_negative_samples(&model, 5);
        model.train(&dataset, 100);
        model.save(model_filename);

        model
            .predict_to_image(webcam1)
            .save("out/test_hard_negative_samples_webcam1.png")
            .unwrap();
        model
            .predict_to_image(loco5)
            .save("out/test_hard_negative_samples_loco5.png")
            .unwrap();
        model
            .predict_to_image(marker1)
            .save("out/test_hard_negative_samples_marker1.png")
            .unwrap();
        model
            .predict_to_image(marker2)
            .save("out/test_hard_negative_samples_marker2.png")
            .unwrap();
    }
}
