use crate::bbox::BBox;
use crate::detection::{nms_sort, Detection};
use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};
use std::io::Cursor;
use tract_ndarray::Axis;
use tract_onnx::prelude::*;

fn sigmoid(a: &f32) -> f32 {
    1.0 / (1.0 + (-a).exp())
}

pub const SIZE: usize = 416;
pub const TINY_YOLOV2_ANCHOR_PRIORS: [f32; 10] = [
    1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52,
];

pub type ModelType = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Clone)]
pub struct Yolo {
    pub model: ModelType,
    pub detections: Vec<Detection>,
}

impl Yolo {
    pub fn model() -> ModelType {
        let data = include_bytes!("../../res/tinyyolov2-7.onnx");
        let mut cursor = Cursor::new(data);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .unwrap()
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, SIZE, SIZE)),
            )
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        model
    }

    pub fn classes() -> Vec<String> {
        let collect = include_str!("../../res/voc.names")
            .to_string()
            .lines()
            .map(|s| s.to_string())
            .collect();
        collect
    }

    pub fn scale(width: u32, height: u32, abox: &BBox) -> Rect {
        let r = abox.scale_to_rect(width as i32, height as i32);
        Rect::at(r.0, r.1).of_size(r.2, r.3)
    }

    pub fn detect_objects(
        &self,
        image: &Box<DynamicImage>,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        let image_rgb = image.to_rgb8();
        let resized =
            image::imageops::resize(&image_rgb, 416, 416, ::image::imageops::FilterType::Nearest);
        let tensor: Tensor =
            tract_ndarray::Array4::from_shape_fn((1, 3, SIZE, SIZE), |(_, c, y, x)| {
                resized[(x as _, y as _)][c] as f32
            })
            .into();
        let result = self.model.run(tvec!(tensor)).unwrap();
        let result: tract_ndarray::ArrayView4<f32> =
            result[0].to_array_view::<f32>()?.into_dimensionality()?;
        let result = result.index_axis(Axis(0), 0);

        let threshold = 0.5;
        let num_classes = Self::classes().len();

        let mut detections: Vec<Detection> = Vec::new();

        for (cy, iy) in result.axis_iter(Axis(1)).enumerate() {
            for (cx, ix) in iy.axis_iter(Axis(1)).enumerate() {
                let d = ix;
                for b in 0..5 {
                    let channel = b * (num_classes + 5);
                    let tx = d[channel + 0];
                    let ty = d[channel + 1];
                    let tw = d[channel + 2];
                    let th = d[channel + 3];
                    let tc = d[channel + 4];

                    let x = (cx as f32 + sigmoid(&tx)) * 32.0 / SIZE as f32;
                    let y = (cy as f32 + sigmoid(&ty)) * 32.0 / SIZE as f32;

                    let w = tw.exp() * (TINY_YOLOV2_ANCHOR_PRIORS[b * 2]) * 32.0 / SIZE as f32;
                    let h = th.exp() * (TINY_YOLOV2_ANCHOR_PRIORS[b * 2 + 1]) * 32.0 / SIZE as f32;

                    let tc = sigmoid(&tc);
                    let mut max_prob = (0, 0.0);
                    for c in 0..(num_classes) {
                        let v = d[5 + c] * tc;
                        if v > max_prob.1 {
                            max_prob = (c, v);
                        }
                    }
                    if max_prob.1 > threshold {
                        let bbox = BBox {
                            x: x as f64,
                            y: y as f64,
                            w: w as f64,
                            h: h as f64,
                        };
                        detections.push(Detection {
                            class: max_prob.0,
                            bbox,
                            confidence: max_prob.1 * tc,
                        });
                    }
                }
            }
        }
        let detections = nms_sort(detections);
        Ok(detections)
    }

    pub fn run(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = (image.width(), image.height());
        let detections = self.detect_objects(&Box::new(image.clone()));
        let mut img_copy = image.to_rgba8();
        let color = Rgba([125u8, 255u8, 0u8, 0u8]);
        for detection in detections.unwrap() {
            let r = Yolo::scale(width, height, &detection.bbox);
            draw_hollow_rect_mut(&mut img_copy, r, color);
            let font_data = include_bytes!("../../res/Arial.ttf");
            let font = Font::try_from_bytes(font_data as &[u8]).unwrap();

            const FONT_SCALE: f32 = 10.0;
            let label = Yolo::classes()[detection.class].to_string();

            draw_text_mut(
                &mut img_copy,
                Rgba([125u8, 255u8, 0u8, 0u8]),
                r.left() as u32,
                r.top() as u32,
                Scale::uniform(FONT_SCALE),
                &font,
                &format!("#{}", label),
            );
        }
        DynamicImage::ImageRgba8(img_copy)
    }
}

impl Default for Yolo {
    fn default() -> Self {
        Yolo {
            model: Self::model(),
            detections: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn default() {
        let yolo = Yolo::default();
        let img = Box::new(image::open("res/webcam06.jpg").unwrap());
        let detections = yolo.detect_objects(&img);
        for detection in detections.unwrap() {
            let r = Yolo::scale(img.width(), img.height(), &detection.bbox);
            println!("{:?} {:?}", Yolo::classes()[detection.class - 1], r);
        }
    }
}
