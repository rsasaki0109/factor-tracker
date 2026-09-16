//! Association costs and gating rules shared by the trackers.
//!
//! The two trackers use different cost functions because their observations
//! live in different spaces and have different units. Costs are never summed
//! across units; within one tracker a single, well-defined cost is used.

use crate::types::{BBox2D, ClassLabel};

/// Class compatibility gate.
///
/// Two *known* and different classes are incompatible. A missing (`None`)
/// class is treated as compatible with everything.
pub fn classes_compatible(a: Option<ClassLabel>, b: Option<ClassLabel>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// IoU association cost `1 - IoU`.
///
/// Returns `None` when the pair is gated out because `iou < min_iou`. The cost
/// is dimensionless and lies in `[0, 1]`.
pub fn iou_association_cost(predicted: &BBox2D, observed: &BBox2D, min_iou: f64) -> Option<f64> {
    let iou = predicted.iou(observed);
    if iou < min_iou {
        None
    } else {
        Some(1.0 - iou)
    }
}

/// Squared Mahalanobis gating cost.
///
/// Returns `None` when `squared_distance > max_squared_distance`. The cost is
/// in squared standard deviations and is defined for the full measurement
/// dimension (3 for a 3D position).
pub fn mahalanobis_association_cost(
    squared_distance: f64,
    max_squared_distance: f64,
) -> Option<f64> {
    if squared_distance > max_squared_distance {
        None
    } else {
        Some(squared_distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_gate_rules() {
        let car = Some(ClassLabel::new(1));
        let person = Some(ClassLabel::new(2));
        assert!(classes_compatible(car, car));
        assert!(classes_compatible(car, None));
        assert!(classes_compatible(None, person));
        assert!(!classes_compatible(car, person));
    }

    #[test]
    fn iou_cost_and_gate() {
        let a = BBox2D::from_corners(0.0, 0.0, 10.0, 10.0);
        let b = BBox2D::from_corners(0.0, 0.0, 10.0, 10.0);
        assert_eq!(iou_association_cost(&a, &b, 0.5), Some(0.0));

        let c = BBox2D::from_corners(100.0, 100.0, 110.0, 110.0);
        assert_eq!(iou_association_cost(&a, &c, 0.1), None);
    }

    #[test]
    fn mahalanobis_gate() {
        assert_eq!(mahalanobis_association_cost(4.0, 9.0), Some(4.0));
        assert_eq!(mahalanobis_association_cost(9.5, 9.0), None);
    }
}
