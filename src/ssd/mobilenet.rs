use super::predictable::{detect_objects, windows, Predictable};
use crate::detection::Detection;
use image::DynamicImage;
use image::{RgbImage, Rgba};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};
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

    fn predict_image(&self, image: RgbImage) -> Vec<u32> {
        let (_cols, _rows, windows) = windows(&image, self.input_width as u32);
        let windows = windows.iter().map(|window| window.to_image()).collect();
        self.predict(windows)
    }

    fn predict_to_image(&self, image: RgbImage) -> DynamicImage {
        let window_size = self.input_width as u32;
        let cols = 2 * (image.width() / window_size);
        let rows = 2 * (image.height() / window_size);
        let predictions = self.predict_image(image.clone());
        let detections = detect_objects(cols, rows, predictions, window_size);

        let mut img_copy = DynamicImage::ImageRgb8(image).to_rgba8();
        for detection in detections.iter() {
            let color = Rgba([125u8, 255u8, 0u8, 0u8]);
            draw_cross_mut(
                &mut img_copy,
                Rgba([255u8, 0u8, 0u8, 0u8]),
                detection.bbox.x as i32,
                detection.bbox.y as i32,
            );
            draw_hollow_rect_mut(
                &mut img_copy,
                Rect::at(
                    (detection.bbox.x as u32) as i32,
                    (detection.bbox.y as u32) as i32,
                )
                .of_size(window_size, window_size),
                color,
            );

            let font_data = include_bytes!("../../res/Arial.ttf");
            let font = Font::try_from_bytes(font_data as &[u8]).unwrap();

            const FONT_SCALE: f32 = 10.0;

            draw_text_mut(
                &mut img_copy,
                Rgba([125u8, 255u8, 0u8, 0u8]),
                detection.bbox.x as u32,
                detection.bbox.y as u32,
                Scale::uniform(FONT_SCALE),
                &font,
                &format!("{}", detection.class),
            );
        }
        DynamicImage::ImageRgba8(img_copy)
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
        assert_eq!(model.input_height, 96);
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

    #[test]
    fn test_predict_to_image() {
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();

        let model = MobileNet::default();

        model
            .predict_to_image(webcam1)
            .save("out/test_predict_to_image_webcam1.png")
            .unwrap();
    }
}
