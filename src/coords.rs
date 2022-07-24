use nalgebra::{Affine2, Matrix3, Point2};
use num::NumCast;

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
            p1: Point2::new(num::cast(1000.0).unwrap(), num::cast(0.0).unwrap()),
            p2: Point2::new(num::cast(0.0).unwrap(), num::cast(0.0).unwrap()),
            p3: Point2::new(num::cast(0.0).unwrap(), num::cast(1000.0).unwrap()),
            q1: Point2::new(num::cast(100.0).unwrap(), num::cast(0.0).unwrap()),
            q2: Point2::new(num::cast(0.0).unwrap(), num::cast(0.0).unwrap()),
            q3: Point2::new(num::cast(0.0).unwrap(), num::cast(100.0).unwrap()),
            affine: Affine2::identity(),
        };
        coords.update_affine();
        coords
    }
}

impl<T: nalgebra::RealField + NumCast + Copy> Coords<T> {
    pub fn new(p1: Point2<T>, p2: Point2<T>, p3: Point2<T>) -> Coords<T> {
        let q1: Point2<T> = Point2::new(num::cast(100.0).unwrap(), num::cast(0.0).unwrap());
        let q2 = Point2::new(num::cast(0.0).unwrap(), num::cast(0.0).unwrap());
        let q3 = Point2::new(num::cast(0.0).unwrap(), num::cast(100.0).unwrap());
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
            self.p1.y,
            num::cast(1.0).unwrap(),
            self.p2.y,
            self.p2.x,
            num::cast(1.0).unwrap(),
            self.p3.x,
            self.p3.y,
            num::cast(1.0).unwrap(),
        );
        let q = Matrix3::new(
            self.q1.x,
            self.q1.y,
            num::cast(1.0).unwrap(),
            self.q2.y,
            self.q2.x,
            num::cast(1.0).unwrap(),
            self.q3.x,
            self.q3.y,
            num::cast(1.0).unwrap(),
        );
        let t = p * q.try_inverse().unwrap();

        println!("{:?}", self.affine);
        let affine = Affine2::from_matrix_unchecked(t);

        println!("{:?}", affine);
        self.affine = affine;
    }

    pub fn set_point1(&mut self, p1: Point2<T>, q1: Point2<T>) {
        self.p1 = p1;
        self.q1 = q1;
    }

    pub fn set_point2(&mut self, p2: Point2<T>, q2: Point2<T>) {
        self.p2 = p2;
        self.q2 = q2;
    }

    pub fn set_point3(&mut self, p3: Point2<T>, q3: Point2<T>) {
        self.p3 = p3;
        self.q3 = q3;
    }

    pub fn to_coord1(&self, point: &Point2<T>) -> Point2<T> {
        self.affine.transform_point(&point)
    }

    pub fn to_coord2(&self, point: &Point2<T>) -> Point2<T> {
        self.affine.inverse_transform_point(&point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default() {
        let coords = Coords::default();
        let p1 = Point2::new(1000.0, 1000.0);
        assert_ne!(coords.to_coord1(&p1), coords.to_coord2(&p1));
        assert_eq!(Point2::new(1000.0, 1000.0), coords.to_coord2(&p1));
    }
}
