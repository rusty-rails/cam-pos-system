use super::predictable::Predictable;
use crate::bbox::BBox;
use crate::detection::Detection;
use image::RgbImage;
use image::{DynamicImage, GenericImageView};
use imageproc::rect::Rect;
use std::io::Cursor;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::dims;

pub const SIZE: usize = 96;

pub type ModelType = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Clone)]
pub struct MobileNet {
    pub model: ModelType,
    pub detections: Vec<Detection>,
    pub input_width: usize,
    pub input_height: usize,
}

impl MobileNet {
    pub fn model() -> ModelType {
        let data = include_bytes!("../../res/markers_n_train_mobilenetv2.onnx");
        let mut cursor = Cursor::new(data);
        let batch = Symbol::new('N');
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .unwrap()
            .with_input_fact(0, f32::fact(dims!(batch, SIZE, SIZE, 3)).into())
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        model
    }

    pub fn classes() -> Vec<String> {
        let collect = include_str!("../../res/labels.txt")
            .to_string()
            .lines()
            .map(|s| s.to_string())
            .collect();
        collect
    }

    pub fn detect_class(
        &self,
        image: &DynamicImage,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let image_rgb = image.to_rgb8();
        let resized = image::imageops::resize(
            &image_rgb,
            self.input_width as u32,
            self.input_height as u32,
            ::image::imageops::FilterType::Triangle,
        );
        let tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
            (1, self.input_height, self.input_width, 3),
            |(_, y, x, c)| resized[(x as _, y as _)][c] as f32 / 255.0,
        )
        .into();

        let result = self.model.run(tvec!(tensor)).unwrap();
        let best = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(0..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let index = best.unwrap().1;
        Ok(Some(index))
    }

    pub fn scale(width: u32, height: u32, abox: &BBox) -> Rect {
        let r = abox.scale_to_rect(width as i32, height as i32);
        Rect::at(r.0, r.1).of_size(r.2, r.3)
    }

    pub fn detect_objects(
        &self,
        image: &Box<DynamicImage>,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        let mut detections: Vec<Detection> = Vec::new();
        Ok(detections)
    }

    pub fn run(&self, image: &DynamicImage) -> DynamicImage {
        let mut img_copy = image.to_rgba8();
        DynamicImage::ImageRgba8(img_copy)
    }
}

impl Default for MobileNet {
    fn default() -> Self {
        MobileNet {
            model: Self::model(),
            detections: Vec::new(),
            input_width: 96,
            input_height: 96,
        }
    }
}

impl Predictable for MobileNet {
    fn predict(&self, windows: Vec<RgbImage>) -> Vec<u32> {
        let resized: Vec<RgbImage> = windows
            .iter()
            .map(|window| {
                image::imageops::resize(
                    window,
                    self.input_width as u32,
                    self.input_height as u32,
                    ::image::imageops::FilterType::Triangle,
                )
            })
            .collect();
        let n = resized.len();
        let tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
            (n, self.input_height, self.input_width, 3),
            |(n, y, x, c)| resized[n][(x as _, y as _)][c] as f32 / 255.0,
        )
        .into_tensor();
        let result = self.model.run(tvec!(tensor)).unwrap();

        let best: tract_ndarray::ArrayView2<f32> = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .into_dimensionality()
            .unwrap();
        let mut result = Vec::new();
        for (_ix, b) in best.outer_iter().enumerate() {
            let best = b
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            result.push(best.0 as u32);
        }
        result
    }

    fn predict_image(&self, window: RgbImage) -> Vec<u32> {
        let resized = image::imageops::resize(
            &window,
            self.input_width as u32,
            self.input_height as u32,
            ::image::imageops::FilterType::Triangle,
        );
        let tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
            (1, self.input_height, self.input_width, 3),
            |(_, y, x, c)| resized[(x as _, y as _)][c] as f32 / 255.0,
        )
        .into();

        let result = self.model.run(tvec!(tensor)).unwrap();
        let best = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .zip(0..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let index = best.unwrap().1;
        vec![index]
    }
    fn predict_to_image(&self, image: RgbImage) -> DynamicImage {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::super::window_crop;
    use super::*;
    use image::open;

    #[test]
    fn default() {
        let model = MobileNet::default();
        let img = Box::new(image::open("res/webcam06.jpg").unwrap());
        let detections = model.detect_objects(&img);
    }

    #[test]
    fn test_classes() {
        assert_eq!(MobileNet::classes()[0], "none");
    }

    #[test]
    fn test_detect_class() {
        let window_size = 28;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (280, 370));
        let marker1 = window_crop(&webcam1, window_size, window_size, (540, 90));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 25));

        let model = MobileNet::default();

        assert_eq!(
            model
                .detect_class(&DynamicImage::ImageRgb8(marker1))
                .unwrap(),
            Some(1)
        );
        assert_eq!(
            model
                .detect_class(&DynamicImage::ImageRgb8(marker2))
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            model.detect_class(&DynamicImage::ImageRgb8(loco5)).unwrap(),
            Some(5)
        );
    }

    #[test]
    fn test_predict() {
        let window_size = 28;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (280, 370));
        let marker1 = window_crop(&webcam1, window_size, window_size, (540, 90));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 25));
        let images = vec![loco5, marker1, marker2];

        let model = MobileNet::default();

        assert_eq!(model.predict(images), vec![5, 1, 2]);
    }

    #[test]
    fn test_predict_image() {
        let window_size = 28;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (280, 370));
        let marker1 = window_crop(&webcam1, window_size, window_size, (540, 90));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 25));

        let model = MobileNet::default();

        assert_eq!(model.predict_image(marker1)[0], 1);
        assert_eq!(
            model
                .detect_class(&DynamicImage::ImageRgb8(marker2))
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            model.detect_class(&DynamicImage::ImageRgb8(loco5)).unwrap(),
            Some(5)
        );
    }
}
