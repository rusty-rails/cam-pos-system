use image::{GrayImage, ImageBuffer, Luma};
use imageproc::template_matching::{find_extremes, match_template, Extremes, MatchTemplateMethod};
use nalgebra::Point2;

pub struct ObjectFinder {
    templates: Vec<GrayImage>,
}

impl ObjectFinder {
    pub fn new(templates: Vec<GrayImage>) -> Self {
        ObjectFinder { templates }
    }

    fn run_match_template(&self, image: &GrayImage, method: MatchTemplateMethod) -> Extremes<f32> {
        let mut summed = ImageBuffer::<Luma<f32>, Vec<f32>>::new(image.width(), image.height());

        for template in &self.templates {
            let result = match_template(&image, template, method);
            for (p, q) in summed.pixels_mut().zip(result.pixels()) {
                p.0[0] = p.0[0] + q.0[0];
            }
            println!("{:?}", find_extremes(&summed));
        }

        find_extremes(&summed)
    }

    pub fn find(&self, image: &GrayImage) -> Point2<f32> {
        let extremes =
            self.run_match_template(image, MatchTemplateMethod::SumOfSquaredErrorsNormalized);
        Point2::new(
            extremes.min_value_location.0 as f32,
            extremes.min_value_location.1 as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[ignore]
    #[test]
    fn default() {
        use image::open;
        let loco01 = open("res/loco01.jpg").unwrap().to_luma8();
        let loco02 = open("res/loco02.jpg").unwrap().to_luma8();
        let loco03 = open("res/loco03.jpg").unwrap().to_luma8();
        let loco04 = open("res/loco04.jpg").unwrap().to_luma8();
        let loco05 = open("res/loco05.jpg").unwrap().to_luma8();

        let templates = vec![loco01, loco02, loco03, loco04, loco05];

        let object_finder = ObjectFinder::new(templates);

        let webcam1 = open("res/webcam01.jpg").unwrap().to_luma8();
        let webcam6 = open("res/webcam06.jpg").unwrap().to_luma8();

        assert_relative_eq!(
            Point2::new(300.0, 500.0),
            object_finder.find(&webcam1),
            epsilon = 500.1
        );
        assert_relative_eq!(
            Point2::new(100.0, 100.0),
            object_finder.find(&webcam6),
            epsilon = 50.0
        );
    }
}
