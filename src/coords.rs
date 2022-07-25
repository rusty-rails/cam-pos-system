use nalgebra::{Affine2, Matrix3, Point2};
use num::NumCast;

// p is real world, q is model world
pub struct Coords<T: nalgebra::RealField + Copy> {
    p1: Point2<T>,
    p2: Point2<T>,
    p3: Point2<T>,
    q1: Point2<T>,
    q2: Point2<T>,
    q3: Point2<T>,
    affine: Affine2<T>,
}

impl<T: nalgebra::RealField + NumCast + Copy> Default for Coords<T> {
    fn default() -> Self {
        let mut coords = Coords {
            p1: Point2::new(num::cast(100.0).unwrap(), num::cast(0.0).unwrap()),
            p2: Point2::new(num::cast(0.0).unwrap(), num::cast(0.0).unwrap()),
            p3: Point2::new(num::cast(0.0).unwrap(), num::cast(100.0).unwrap()),
            q1: Point2::new(num::cast(1000.0).unwrap(), num::cast(0.0).unwrap()),
            q2: Point2::new(num::cast(0.0).unwrap(), num::cast(0.0).unwrap()),
            q3: Point2::new(num::cast(0.0).unwrap(), num::cast(1000.0).unwrap()),
            affine: Affine2::identity(),
        };
        coords.update_affine();
        coords
    }
}

impl<T: nalgebra::RealField + NumCast + Copy> Coords<T> {
    pub fn new(p1: Point2<T>, p2: Point2<T>, p3: Point2<T>) -> Coords<T> {
        let q1: Point2<T> = Point2::new(num::cast(1000.0).unwrap(), num::cast(0.0).unwrap());
        let q2 = Point2::new(num::cast(0.0).unwrap(), num::cast(0.0).unwrap());
        let q3 = Point2::new(num::cast(0.0).unwrap(), num::cast(1000.0).unwrap());
        let mut coords = Coords {
            p1,
            p2,
            p3,
            q1,
            q2,
            q3,
            affine: Affine2::identity(),
        };
        coords.update_affine();
        coords
    }

    // https://math.stackexchange.com/questions/1092002/how-to-define-an-affine-transformation-using-2-triangles
    // https://github.com/chrvadala/transformation-matrix/blob/main/src/fromTriangles.js
    pub fn update_affine(&mut self) {
        let p = Matrix3::new(
            self.p1.x,
            self.p2.x,
            self.p3.x,
            self.p1.y,
            self.p2.y,
            self.p3.y,
            num::cast(1.0).unwrap(),
            num::cast(1.0).unwrap(),
            num::cast(1.0).unwrap(),
        );
        let q = Matrix3::new(
            self.q1.x,
            self.q2.x,
            self.q3.x,
            self.q1.y,
            self.q2.y,
            self.q3.y,
            num::cast(1.0).unwrap(),
            num::cast(1.0).unwrap(),
            num::cast(1.0).unwrap(),
        );
        let t = p * q.try_inverse().unwrap();
        let affine = Affine2::from_matrix_unchecked(t);
        self.affine = affine;
    }

    pub fn set_point1(&mut self, p1: Point2<T>, q1: Point2<T>) {
        self.p1 = p1;
        self.q1 = q1;
        self.update_affine();
    }

    pub fn set_point2(&mut self, p2: Point2<T>, q2: Point2<T>) {
        self.p2 = p2;
        self.q2 = q2;
        self.update_affine();
    }

    pub fn set_point3(&mut self, p3: Point2<T>, q3: Point2<T>) {
        self.p3 = p3;
        self.q3 = q3;
        self.update_affine();
    }

    pub fn set_marker1(&mut self, marker1: Point2<T>) {
        self.p1 = marker1;
        self.update_affine();
    }

    pub fn set_marker2(&mut self, marker2: Point2<T>) {
        self.p2 = marker2;
        self.update_affine();
    }

    pub fn set_marker3(&mut self, marker3: Point2<T>) {
        self.p3 = marker3;
        self.update_affine();
    }

    pub fn to_model(&self, point: &Point2<T>) -> Point2<T> {
        self.affine.inverse_transform_point(&point)
    }

    pub fn to_world(&self, point: &Point2<T>) -> Point2<T> {
        self.affine.transform_point(&point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn default() {
        let coords = Coords::default();
        let p = Point2::new(100.0, 100.0);
        assert_ne!(coords.to_model(&p), coords.to_world(&p));

        assert_relative_eq!(
            Point2::new(1000.0, 1000.0),
            coords.to_model(&p),
            epsilon = 0.001
        );
        assert_relative_eq!(
            Point2::new(10.0, 10.0),
            coords.to_world(&p),
            epsilon = 0.001
        );
    }

    #[test]
    fn set_marker() {
        let mut coords = Coords::default();
        let marker1 = Point2::new(100.0, 50.0);
        let marker2 = Point2::new(0.0, 50.0);
        let point = Point2::new(50.0, 50.0);

        coords.set_marker1(marker1);
        coords.set_marker2(marker2);

        assert_relative_eq!(
            Point2::new(500.0, 0.0),
            coords.to_model(&point),
            epsilon = 0.001
        );

        let point2 = Point2::new(0.0, 0.0);
        assert_relative_eq!(
            Point2::new(0.0, -1000.0),
            coords.to_model(&point2),
            epsilon = 0.001
        );
    }
}
