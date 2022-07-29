// source: https://github.com/12101111/yolo-rs/blob/master/src/yolo.rs
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl BBox {
    pub fn left(&self) -> f64 {
        self.x - self.w / 2.0
    }
    pub fn right(&self) -> f64 {
        self.x + self.w / 2.0
    }
    pub fn top(&self) -> f64 {
        self.y - self.h / 2.0
    }
    pub fn bot(&self) -> f64 {
        self.y + self.h / 2.0
    }
    pub fn overlay(&self, rhs: &BBox) -> f64 {
        let left = self.left().max(rhs.left());
        let right = self.right().min(rhs.right());
        let w = (right - left).max(0.0);
        let top = self.top().max(rhs.top());
        let bot = self.bot().min(rhs.bot());
        let h = (bot - top).max(0.0);
        w * h
    }
    pub fn union(&self, rhs: &BBox) -> f64 {
        self.w * self.h + rhs.w * rhs.h - self.overlay(rhs)
    }
    pub fn iou(&self, rhs: &BBox) -> f64 {
        self.overlay(rhs) / self.union(rhs)
    }
    pub fn scale_to_rect(&self, imw: i32, imh: i32) -> (i32, i32, u32, u32) {
        let w = imw as f64;
        let h = imh as f64;
        let left = ((self.left() * w) as i32).max(0);
        let right = ((self.right() * w) as i32).min(imw - 1);
        let top = ((self.top() * h) as i32).max(0);
        let bot = ((self.bot() * h) as i32).min(imh - 1);
        (left, top, (right - left) as u32, (bot - top) as u32)
    }
}

#[test]
pub fn iou() {
    let b1 = BBox {
        x: 0.5,
        y: 0.5,
        w: 1.0,
        h: 1.0,
    };
    assert_eq!(b1.left(), 0.0);
    assert_eq!(b1.right(), 1.0);
    assert_eq!(b1.top(), 0.0);
    assert_eq!(b1.bot(), 1.0);
    assert_eq!(b1.overlay(&b1), 1.0);
    assert_eq!(b1.union(&b1), 1.0);
    assert_eq!(b1.iou(&b1), 1.0);
}
