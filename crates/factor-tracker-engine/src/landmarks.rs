//! Ego-pose correction from static landmarks.
//!
//! Dataset and odometry poses jitter. On two of the CODa evaluation windows
//! static objects move by 0.2-0.3 m between scans in the world frame, and
//! every track inherits those jumps. [`LandmarkPoseCorrector`] keeps a map of
//! static objects (poles, trees, bins: whatever the detector reports as
//! static) in the tracking frame and, for each scan, finds the horizontal
//! rigid correction (a rotation about z plus a translation) that best aligns
//! the scan's static detections with the map. The engine applies it to the
//! scan's detections before tracking, so consecutive scans share one frame.
//!
//! Per scan, starting from the previous scan's correction `C`:
//!
//! ```text
//! 1. x_i = C(p_i)                 the scan's landmark detections, tracking frame
//! 2. match each x_i to the nearest confirmed map landmark m_j within match_radius_m
//! 3. C <- argmin_D sum_i huber(|D(x_i) - m_j|) ∘ C      (twice, re-matching after the first)
//! 4. update the map with C(p_i): running mean of the landmark each one matches,
//!    a new landmark otherwise; forget landmarks not seen for forget_after_ns
//! ```
//!
//! Step 3 is the optimum of a one-scan factor graph: one pose-correction
//! variable and a robust position factor per matched landmark, solved in
//! closed form (weighted Procrustes, re-weighted for the Huber loss). The
//! correction only changes when at least `min_matches` landmarks match, so a
//! scan without landmarks keeps the previous one.

use factor_tracker_core::{RigidTransform3D, Timestamp};
use nalgebra::{Matrix3, Vector2, Vector3};

use crate::error::{EngineError, EngineResult};

/// Configuration of [`LandmarkPoseCorrector`].
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkConfig {
    /// A landmark detection matches a map landmark within this horizontal
    /// distance (m).
    pub match_radius_m: f64,
    /// Sightings (scans) before a map landmark is used for alignment.
    pub confirm_sightings: u32,
    /// Matched landmarks needed to update the correction.
    pub min_matches: usize,
    /// Residual (m) above which the alignment down-weights a match (Huber).
    pub huber_m: f64,
    /// A map landmark not seen for this long (ns) is forgotten.
    pub forget_after_ns: i64,
    /// A map landmark's position is the running mean of at most this many
    /// sightings, so it can follow slow drift.
    pub mean_window: u32,
}

impl Default for LandmarkConfig {
    fn default() -> Self {
        // Tuned on the five CODa development windows; HOTA moves by at most
        // 0.002 over 0.5-1.0 m radius, 2-5 sightings, 2-20 s memory and a
        // 10-200 sighting window (docs/evaluation.md).
        LandmarkConfig {
            match_radius_m: 0.7,
            confirm_sightings: 3,
            min_matches: 3,
            huber_m: 0.15,
            forget_after_ns: 5_000_000_000,
            mean_window: 50,
        }
    }
}

impl LandmarkConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> EngineResult<()> {
        let positive = [
            ("match_radius_m", self.match_radius_m),
            ("huber_m", self.huber_m),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(EngineError::config(format!(
                    "pose correction: {name} must be > 0"
                )));
            }
        }
        if self.min_matches < 2 {
            return Err(EngineError::config(
                "pose correction: min_matches must be >= 2",
            ));
        }
        if self.confirm_sightings == 0 || self.mean_window == 0 || self.forget_after_ns <= 0 {
            return Err(EngineError::config(
                "pose correction: confirm_sightings, mean_window and forget_after_ns must be > 0",
            ));
        }
        Ok(())
    }
}

/// A horizontal rigid transform: rotation `theta` about z, then translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid2D {
    /// Rotation about z (rad).
    pub theta: f64,
    /// Translation in x and y (m).
    pub translation: Vector2<f64>,
}

impl Rigid2D {
    /// The identity.
    pub const IDENTITY: Rigid2D = Rigid2D {
        theta: 0.0,
        translation: Vector2::new(0.0, 0.0),
    };

    fn rotate(&self, p: &Vector2<f64>) -> Vector2<f64> {
        let (s, c) = self.theta.sin_cos();
        Vector2::new(c * p.x - s * p.y, s * p.x + c * p.y)
    }

    /// Applies the transform to a horizontal position.
    pub fn apply(&self, p: &Vector2<f64>) -> Vector2<f64> {
        self.rotate(p) + self.translation
    }

    /// Returns `self ∘ other` (apply `other` first).
    pub fn compose(&self, other: &Rigid2D) -> Rigid2D {
        Rigid2D {
            theta: self.theta + other.theta,
            translation: self.rotate(&other.translation) + self.translation,
        }
    }

    /// The same transform in 3D (z is unchanged).
    pub fn to_rigid3d(&self) -> RigidTransform3D {
        let (s, c) = self.theta.sin_cos();
        RigidTransform3D::new(
            Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
            Vector3::new(self.translation.x, self.translation.y, 0.0),
        )
        .expect("a rotation about z is a valid rotation")
    }
}

#[derive(Debug, Clone)]
struct Landmark {
    position: Vector2<f64>,
    sightings: u32,
    last_seen: Timestamp,
}

/// Per-scan ego-pose correction from static landmarks. See the module docs.
#[derive(Debug, Clone)]
pub struct LandmarkPoseCorrector {
    config: LandmarkConfig,
    landmarks: Vec<Landmark>,
    correction: Rigid2D,
}

impl LandmarkPoseCorrector {
    /// Creates a corrector with an empty map.
    pub fn new(config: LandmarkConfig) -> EngineResult<Self> {
        config.validate()?;
        Ok(LandmarkPoseCorrector {
            config,
            landmarks: Vec::new(),
            correction: Rigid2D::IDENTITY,
        })
    }

    /// Configuration.
    pub fn config(&self) -> &LandmarkConfig {
        &self.config
    }

    /// The latest correction (identity before any scan).
    pub fn correction(&self) -> Rigid2D {
        self.correction
    }

    /// Number of map landmarks.
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    /// Processes one scan's landmark detections (tracking frame, before
    /// correction) and returns the
    /// correction for that scan with the number of landmarks it was aligned
    /// on (0 when it kept the previous correction).
    pub fn correct(
        &mut self,
        timestamp: Timestamp,
        detections: &[Vector3<f64>],
    ) -> (Rigid2D, usize) {
        let raw: Vec<Vector2<f64>> = detections.iter().map(|p| p.xy()).collect();
        let radius = self.config.match_radius_m;
        let confirmed: Vec<Vector2<f64>> = self
            .landmarks
            .iter()
            .filter(|l| l.sightings >= self.config.confirm_sightings)
            .map(|l| l.position)
            .collect();
        let mut correction = self.correction;
        let mut matched = 0;
        if confirmed.len() >= self.config.min_matches {
            for _ in 0..2 {
                let (from, to): (Vec<_>, Vec<_>) = raw
                    .iter()
                    .filter_map(|p| {
                        let x = correction.apply(p);
                        nearest(&confirmed, &x, radius).map(|j| (x, confirmed[j]))
                    })
                    .unzip();
                if from.len() < self.config.min_matches {
                    break;
                }
                matched = from.len();
                correction = robust_alignment(&from, &to, self.config.huber_m).compose(&correction);
            }
        }
        self.correction = correction;
        self.update_map(timestamp, &raw);
        (correction, matched)
    }

    fn update_map(&mut self, timestamp: Timestamp, raw: &[Vector2<f64>]) {
        let radius = self.config.match_radius_m;
        let mut taken = vec![false; self.landmarks.len()];
        for p in raw {
            let x = self.correction.apply(p);
            let best = self
                .landmarks
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken[*i])
                .map(|(i, l)| (i, (l.position - x).norm()))
                .filter(|&(_, d)| d < radius)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, _)) => {
                    let landmark = &mut self.landmarks[i];
                    landmark.sightings += 1;
                    let n = landmark.sightings.min(self.config.mean_window) as f64;
                    landmark.position += (x - landmark.position) / n;
                    landmark.last_seen = timestamp;
                    taken[i] = true;
                }
                None => {
                    self.landmarks.push(Landmark {
                        position: x,
                        sightings: 1,
                        last_seen: timestamp,
                    });
                    taken.push(true);
                }
            }
        }
        let forget = self.config.forget_after_ns;
        self.landmarks
            .retain(|l| timestamp.as_nanos() - l.last_seen.as_nanos() <= forget);
    }
}

fn nearest(points: &[Vector2<f64>], x: &Vector2<f64>, radius: f64) -> Option<usize> {
    points
        .iter()
        .enumerate()
        .map(|(j, p)| (j, (p - x).norm()))
        .filter(|&(_, d)| d < radius)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(j, _)| j)
}

/// The rigid transform `D` minimising `sum huber(|D(from_i) - to_i|)`:
/// weighted Procrustes, re-weighted five times.
fn robust_alignment(from: &[Vector2<f64>], to: &[Vector2<f64>], huber: f64) -> Rigid2D {
    let mut weights = vec![1.0; from.len()];
    let mut result = Rigid2D::IDENTITY;
    for _ in 0..5 {
        let total: f64 = weights.iter().sum();
        let mean = |points: &[Vector2<f64>]| {
            points
                .iter()
                .zip(&weights)
                .map(|(p, w)| p * *w)
                .sum::<Vector2<f64>>()
                / total
        };
        let (mx, my) = (mean(from), mean(to));
        let (mut dot, mut cross) = (0.0, 0.0);
        for ((p, q), w) in from.iter().zip(to).zip(&weights) {
            let (a, b) = (p - mx, q - my);
            dot += w * (a.x * b.x + a.y * b.y);
            cross += w * (a.x * b.y - a.y * b.x);
        }
        let rotation = Rigid2D {
            theta: cross.atan2(dot),
            translation: Vector2::zeros(),
        };
        result = Rigid2D {
            theta: rotation.theta,
            translation: my - rotation.apply(&mx),
        };
        for ((p, q), w) in from.iter().zip(to).zip(weights.iter_mut()) {
            let r = (result.apply(p) - q).norm();
            *w = if r < huber { 1.0 } else { huber / r };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(0.0, 7.0, 0.0),
            Vector3::new(-6.0, -2.0, 0.0),
            Vector3::new(3.0, -8.0, 0.0),
            Vector3::new(-4.0, 9.0, 0.0),
        ]
    }

    fn moved(points: &[Vector3<f64>], jitter: &Rigid2D) -> Vec<Vector3<f64>> {
        points
            .iter()
            .map(|p| {
                let q = jitter.apply(&p.xy());
                Vector3::new(q.x, q.y, p.z)
            })
            .collect()
    }

    fn at(k: i64) -> Timestamp {
        Timestamp::from_nanos(k * 100_000_000)
    }

    #[test]
    fn alignment_recovers_a_rigid_motion() {
        let from: Vec<_> = scene().iter().map(|p| p.xy()).collect();
        let motion = Rigid2D {
            theta: 0.03,
            translation: Vector2::new(0.2, -0.1),
        };
        let mut to: Vec<_> = from.iter().map(|p| motion.apply(p)).collect();
        to[0] += Vector2::new(2.0, 0.0); // one outlier
        let found = robust_alignment(&from, &to, 0.15);
        // Huber bounds the outlier's pull rather than removing it (weight
        // 0.15 / 2): 0.04 m off here, where least squares is 0.4 m off.
        assert!((found.theta - motion.theta).abs() < 2e-3);
        assert!((found.translation - motion.translation).norm() < 0.05);
    }

    #[test]
    fn removes_pose_jitter_after_the_map_is_built() {
        let mut corrector = LandmarkPoseCorrector::new(LandmarkConfig::default()).unwrap();
        for k in 0..5 {
            let (c, _) = corrector.correct(at(k), &scene());
            assert_eq!(c, Rigid2D::IDENTITY);
        }
        // The pose estimate is now off by 0.3 m and 1 degree: the scan's
        // landmarks appear displaced, and the correction moves them back.
        let jitter = Rigid2D {
            theta: 1f64.to_radians(),
            translation: Vector2::new(0.3, 0.0),
        };
        let (c, matched) = corrector.correct(at(5), &moved(&scene(), &jitter));
        assert_eq!(matched, 5);
        for p in scene() {
            let back = c.apply(&jitter.apply(&p.xy()));
            assert!((back - p.xy()).norm() < 0.01, "{back} vs {}", p.xy());
        }
    }

    #[test]
    fn keeps_the_correction_without_enough_landmarks() {
        let mut corrector = LandmarkPoseCorrector::new(LandmarkConfig::default()).unwrap();
        for k in 0..5 {
            corrector.correct(at(k), &scene());
        }
        let jitter = Rigid2D {
            theta: 0.0,
            translation: Vector2::new(0.2, 0.0),
        };
        let (c, _) = corrector.correct(at(5), &moved(&scene(), &jitter));
        let (kept, matched) = corrector.correct(at(6), &moved(&scene()[..2], &jitter));
        assert_eq!(matched, 0);
        assert_eq!(kept, c);
    }

    #[test]
    fn forgets_landmarks_out_of_sight() {
        let mut corrector = LandmarkPoseCorrector::new(LandmarkConfig::default()).unwrap();
        corrector.correct(at(0), &scene());
        assert_eq!(corrector.landmark_count(), 5);
        corrector.correct(at(60), &[]);
        assert_eq!(corrector.landmark_count(), 0);
    }

    #[test]
    fn compose_applies_the_right_transform_last() {
        let a = Rigid2D {
            theta: 0.5,
            translation: Vector2::new(1.0, 0.0),
        };
        let b = Rigid2D {
            theta: -0.2,
            translation: Vector2::new(0.0, 2.0),
        };
        let p = Vector2::new(0.3, -0.7);
        assert!((a.compose(&b).apply(&p) - a.apply(&b.apply(&p))).norm() < 1e-12);
    }
}
