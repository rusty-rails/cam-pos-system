use image::{DynamicImage, ImageBuffer, Luma, Rgba};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use mosse::{MosseTrackerSettings, MultiMosseTracker};
use rusttype::{Font, Scale};

pub struct Tracker {
    psr_threshold: f32,
    window_size: u32,
    pub tracker: MultiMosseTracker,
}

impl Tracker {
    pub fn new(width: u32, height: u32) -> Tracker {
        let window_size = 64;
        let psr_threshold = 7.0;
        let settings = MosseTrackerSettings {
            window_size,
            width,
            height,
            regularization: 0.002,
            learning_rate: 0.05,
            psr_threshold,
        };
        let desperation_threshold = 4;
        let multi_tracker = MultiMosseTracker::new(settings, desperation_threshold);

        Tracker {
            window_size,
            psr_threshold,
            tracker: multi_tracker,
        }
    }

    pub fn add_targets(&mut self, targets: Vec<(u32, u32)>, image: ImageBuffer<Luma<u8>, Vec<u8>>) {
        for (i, coords) in targets.into_iter().enumerate() {
            self.tracker.add_target(i as u32, coords, &image);
        }
    }

    pub fn next(&mut self, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> DynamicImage {
        let predictions = self.tracker.track(&image);

        let mut img_copy = DynamicImage::ImageLuma8(image.clone()).to_rgba8();
        for (obj_id, pred) in predictions.iter() {
            let mut color = Rgba([125u8, 255u8, 0u8, 0u8]);
            if pred.psr < self.psr_threshold {
                color = Rgba([255u8, 0u8, 0u8, 0u8])
            }
            draw_cross_mut(
                &mut img_copy,
                Rgba([255u8, 0u8, 0u8, 0u8]),
                pred.location.0 as i32,
                pred.location.1 as i32,
            );
            let window_size = self.window_size;
            draw_hollow_rect_mut(
                &mut img_copy,
                Rect::at(
                    pred.location.0.saturating_sub(window_size / 2) as i32,
                    pred.location.1.saturating_sub(window_size / 2) as i32,
                )
                .of_size(window_size, window_size),
                color,
            );

            let font_data = include_bytes!("../res/Arial.ttf");
            let font = Font::try_from_bytes(font_data as &[u8]).unwrap();

            const FONT_SCALE: f32 = 10.0;

            draw_text_mut(
                &mut img_copy,
                Rgba([125u8, 255u8, 0u8, 0u8]),
                pred.location.0 - (window_size / 2),
                pred.location.1 - (window_size / 2),
                Scale::uniform(FONT_SCALE),
                &font,
                &format!("#{}", obj_id),
            );

            draw_text_mut(
                &mut img_copy,
                color,
                pred.location.0 - (window_size / 2),
                pred.location.1 - (window_size / 2) + FONT_SCALE as u32,
                Scale::uniform(FONT_SCALE),
                &font,
                &format!("PSR: {:.2}", pred.psr),
            );
        }
        DynamicImage::ImageRgba8(img_copy)
    }
}
