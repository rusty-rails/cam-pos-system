// source: https://github.com/12101111/yolo-rs/blob/master/src/yolo.rs
// x, y is the upper left corner
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn left(&self) -> f32 {
        self.x
    }
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    pub fn top(&self) -> f32 {
        self.y
    }
    pub fn bot(&self) -> f32 {
        self.y + self.h
    }
    pub fn overlay(&self, rhs: &BBox) -> f32 {
        let left = self.left().max(rhs.left());
        let right = self.right().min(rhs.right());
        let w = (right - left).max(0.0);
        let top = self.top().max(rhs.top());
        let bot = self.bot().min(rhs.bot());
        let h = (bot - top).max(0.0);
        w * h
    }
    pub fn union(&self, rhs: &BBox) -> f32 {
        self.w * self.h + rhs.w * rhs.h - self.overlay(rhs)
    }
    pub fn iou(&self, rhs: &BBox) -> f32 {
        self.overlay(rhs) / self.union(rhs)
    }
    pub fn scale_to_rect(&self, imw: i32, imh: i32) -> (i32, i32, u32, u32) {
        let w = imw as f32;
        let h = imh as f32;
        let left = ((self.left() * w) as i32).max(0);
        let right = ((self.right() * w) as i32).min(imw - 1);
        let top = ((self.top() * h) as i32).max(0);
        let bot = ((self.bot() * h) as i32).min(imh - 1);
        (left, top, (right - left) as u32, (bot - top) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn iou() {
        let b1 = BBox {
            x: 0.5,
            y: 0.5,
            w: 1.0,
            h: 1.0,
        };
        assert_eq!(b1.left(), 0.5);
        assert_eq!(b1.right(), 1.5);
        assert_eq!(b1.top(), 0.5);
        assert_eq!(b1.bot(), 1.5);
        assert_eq!(b1.overlay(&b1), 1.0);
        assert_eq!(b1.union(&b1), 1.0);
        assert_eq!(b1.iou(&b1), 1.0);
    }

    #[test]
    pub fn overlay() {
        let b1 = BBox {
            x: 4.0,
            y: 4.0,
            w: 4.4,
            h: 4.4,
        };
        let b2 = BBox {
            x: 8.0,
            y: 4.0,
            w: 4.4,
            h: 4.4,
        };
        assert!(b1.overlay(&b2) > 1.0);
    }
}
