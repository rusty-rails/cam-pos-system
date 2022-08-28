use crate::bbox::BBox;
use crate::detection::{merge, nms_sort, Detection};
use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

pub fn detect_objects(predictions: Vec<(u32, u32, u32)>, window_size: u32) -> Vec<Detection> {
    let mut detections: Vec<Detection> = Vec::new();
    predictions.iter().for_each(|(x, y, class)| {
        if *class > 0 {
            // add 0.1 to generate an overlap on contacting windows.
            let size = 0.01 + window_size as f32;
            let bbox = BBox {
                x: *x as f32,
                y: *y as f32,
                w: size,
                h: size,
            };
            detections.push(Detection {
                class: *class as usize,
                bbox,
                confidence: 1.0,
            });
        }
    });
    let detections = merge(detections);
    nms_sort(detections)
}

pub fn visualize_detections(detector: &dyn Detector, image: &DynamicImage) -> DynamicImage {
    let detections = detector.detect_objects(image);

    let mut img_copy = image.to_rgba8();
    for detection in detections.iter() {
        let color = Rgba([125u8, 255u8, 0u8, 0u8]);
        draw_hollow_rect_mut(
            &mut img_copy,
            Rect::at(detection.bbox.x as i32, detection.bbox.y as i32)
                .of_size(detection.bbox.w as u32, detection.bbox.h as u32),
            color,
        );

        let font_data = include_bytes!("../res/Arial.ttf");
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

pub trait Detector {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection>;
}
