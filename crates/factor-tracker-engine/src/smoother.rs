//! Fixed-lag smoothing of the engine's LiDAR tracks.
//!
//! The engine's states are filtered: the estimate for a scan uses only the
//! detections up to that scan. [`FixedLagSmoother`] holds the last
//! `lag_frames` frames and emits each frame once it leaves the window, with
//! every track's position, velocity and covariance re-estimated from the
//! detections of the whole window: the marginal of the track's factor graph
//! (constant-velocity motion factors, one position factor per detection) over
//! the window. For this linear-Gaussian model it is the Rauch-Tung-Striebel
//! backward pass over the engine's own filtered states, with the same
//! motion model (`cv_f6`, `cv_q6` with the tracker's acceleration spectral
//! density), so nothing is re-associated and ids, lifecycles and
//! observations are unchanged.
//!
//! A frame is emitted `lag_frames` scans late and marked `smoothed`. On CODa
//! pedestrians a 0.5 s lag cuts the median velocity error by 23-29 % and
//! raises HOTA by 0.005-0.010 (docs/evaluation.md).

use std::collections::{HashMap, VecDeque};

use factor_tracker_core::math::{cv_f6, cv_q6};
use nalgebra::{SMatrix, SVector};

use crate::engine::{LidarFrameOutput, TrackOutput};

type State = SVector<f64, 6>;
type Covariance = SMatrix<f64, 6, 6>;

/// Fixed-lag smoother over engine LiDAR frames. See the module docs.
pub struct FixedLagSmoother {
    lag_frames: usize,
    accel_spectral_density: f64,
    frames: VecDeque<LidarFrameOutput>,
}

fn state_of(track: &TrackOutput) -> Option<(State, Covariance)> {
    if track.position.len() != 3 || track.velocity.len() != 3 || track.covariance.len() != 36 {
        return None;
    }
    let x = State::from_iterator(track.position.iter().chain(&track.velocity).copied());
    let p = Covariance::from_row_slice(&track.covariance);
    Some((x, p))
}

impl FixedLagSmoother {
    /// Creates a smoother; `accel_spectral_density` must be the tracker's.
    pub fn new(lag_frames: usize, accel_spectral_density: f64) -> Self {
        FixedLagSmoother {
            lag_frames,
            accel_spectral_density,
            frames: VecDeque::new(),
        }
    }

    /// Adds the engine's next LiDAR frame; returns the frames that left the
    /// window (oldest first), smoothed.
    pub fn push(&mut self, frame: LidarFrameOutput) -> Vec<LidarFrameOutput> {
        self.frames.push_back(frame);
        let mut out = Vec::new();
        while self.frames.len() > self.lag_frames {
            out.push(self.emit_oldest());
        }
        out
    }

    /// Emits every frame still in the window.
    pub fn finish(&mut self) -> Vec<LidarFrameOutput> {
        let mut out = Vec::new();
        while !self.frames.is_empty() {
            out.push(self.emit_oldest());
        }
        out
    }

    fn emit_oldest(&mut self) -> LidarFrameOutput {
        // Filtered states of every track in the window, by frame.
        let window: Vec<HashMap<u64, (State, Covariance)>> = self
            .frames
            .iter()
            .map(|frame| {
                frame
                    .tracks
                    .iter()
                    .filter_map(|t| state_of(t).map(|s| (t.track_id, s)))
                    .collect()
            })
            .collect();
        let times: Vec<f64> = self
            .frames
            .iter()
            .map(|frame| frame.timestamp_ns as f64 * 1e-9)
            .collect();
        let mut frame = self.frames.pop_front().expect("window is not empty");
        frame.smoothed = self.lag_frames > 0;
        for track in &mut frame.tracks {
            let Some(smoothed) = self.smooth(track.track_id, &window, &times) else {
                continue;
            };
            let (x, p) = smoothed;
            track.position = x.fixed_rows::<3>(0).iter().copied().collect();
            track.velocity = x.fixed_rows::<3>(3).iter().copied().collect();
            track.covariance = p.transpose().iter().copied().collect();
        }
        frame
    }

    /// RTS backward pass from the last consecutive frame holding the track
    /// down to the oldest frame.
    fn smooth(
        &self,
        id: u64,
        window: &[HashMap<u64, (State, Covariance)>],
        times: &[f64],
    ) -> Option<(State, Covariance)> {
        let last = window.iter().take_while(|w| w.contains_key(&id)).count();
        if last == 0 {
            return None;
        }
        let (mut xs, mut ps) = window[last - 1][&id];
        for k in (0..last - 1).rev() {
            let (x, p) = window[k][&id];
            let dt = times[k + 1] - times[k];
            if dt <= 0.0 {
                continue;
            }
            let f = cv_f6(dt);
            let x_pred = f * x;
            let p_pred = f * p * f.transpose() + cv_q6(dt, self.accel_spectral_density);
            let gain = p * f.transpose() * p_pred.cholesky()?.inverse();
            xs = x + gain * (xs - x_pred);
            ps = p + gain * (ps - p_pred) * gain.transpose();
            ps = 0.5 * (ps + ps.transpose());
        }
        Some((xs, ps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TrackingMode;
    use crate::CameraAssistedLidarEngine;
    use factor_tracker_core::Timestamp;

    /// A person walking at 1 m/s along x, detected every 0.1 s with a
    /// deterministic +-0.15 m jitter; returns the engine's frames.
    fn walker() -> Vec<LidarFrameOutput> {
        use crate::geometry::{Calibration, CameraIntrinsics};
        use crate::input::{EngineBatch, LidarBatch, LidarDetection};
        use factor_tracker_core::RigidTransform3D;
        use nalgebra::{Matrix3, Vector3};
        let calibration = Calibration {
            tracking_frame: "world".into(),
            lidar_frame: "velodyne".into(),
            camera_frame: "cam".into(),
            lidar_to_reference: RigidTransform3D::identity(),
            camera_to_reference: RigidTransform3D::identity(),
            intrinsics: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 50.0,
                cy: 50.0,
                skew: 0.0,
                image_width: 100,
                image_height: 100,
            },
            near_plane: 0.1,
        };
        let config = crate::EngineConfig {
            mode: TrackingMode::LidarOnly,
            ..crate::EngineConfig::default()
        };
        let mut engine = CameraAssistedLidarEngine::new(config, calibration, None).unwrap();
        for k in 0..40i64 {
            let t = Timestamp::from_nanos(k * 100_000_000);
            let jitter = if k % 2 == 0 { 0.15 } else { -0.15 };
            let batch = LidarBatch {
                sensor_id: factor_tracker_core::SensorId::new(1),
                sequence: k as u64,
                event_time: t,
                frame_id: "velodyne".into(),
                detections: vec![LidarDetection {
                    position: Vector3::new(0.1 * k as f64, jitter, 0.0),
                    covariance: Matrix3::identity() * 0.04,
                    class_label: None,
                    score: 0.9,
                    shape: None,
                    appearance: None,
                }],
                landmarks: Vec::new(),
            };
            engine.ingest(EngineBatch::Lidar(batch), t);
        }
        engine.flush().unwrap();
        engine.lidar_frames().to_vec()
    }

    fn velocity_error(frames: &[LidarFrameOutput]) -> f64 {
        // From scan 10 on, against the true velocity (1, 0, 0).
        let errors: Vec<f64> = frames[10..30]
            .iter()
            .map(|f| {
                let v = &f.tracks[0].velocity;
                ((v[0] - 1.0).powi(2) + v[1].powi(2)).sqrt()
            })
            .collect();
        errors.iter().sum::<f64>() / errors.len() as f64
    }

    #[test]
    fn smoothing_reduces_velocity_error_and_keeps_ids() {
        let frames = walker();
        let mut smoother = FixedLagSmoother::new(
            5,
            crate::EngineConfig::default()
                .tracker
                .accel_spectral_density,
        );
        let mut smoothed: Vec<_> = frames
            .iter()
            .cloned()
            .flat_map(|f| smoother.push(f))
            .collect();
        smoothed.extend(smoother.finish());
        assert_eq!(smoothed.len(), frames.len());
        for (a, b) in frames.iter().zip(&smoothed) {
            assert_eq!(a.timestamp_ns, b.timestamp_ns);
            let ids =
                |f: &LidarFrameOutput| f.tracks.iter().map(|t| t.track_id).collect::<Vec<_>>();
            assert_eq!(ids(a), ids(b));
        }
        let (filtered, smooth) = (velocity_error(&frames), velocity_error(&smoothed));
        assert!(
            smooth < 0.7 * filtered,
            "smoothed {smooth} vs filtered {filtered}"
        );
    }

    #[test]
    fn lag_zero_returns_the_filtered_frames() {
        let frames = walker();
        let mut smoother = FixedLagSmoother::new(0, 1.0);
        let out: Vec<_> = frames
            .iter()
            .cloned()
            .flat_map(|f| smoother.push(f))
            .collect();
        assert_eq!(out, frames);
    }
}
