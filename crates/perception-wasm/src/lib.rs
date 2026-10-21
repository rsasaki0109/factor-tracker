//! Camera-only person tracking for the browser demo (`web/`).
//!
//! [`CameraTracker`] wraps [`BBox2DTracker`] for one video stream: boxes in
//! video pixels, timestamps in milliseconds from `performance.now()`. The
//! `pe_*` functions export it from a `wasm32-unknown-unknown` build with
//! scalar arguments only, so the page needs no generated glue code:
//!
//! ```text
//! pe_reset()
//! per frame: pe_begin(t_ms); pe_add(cx, cy, w, h, score) per detection;
//!            n = pe_end(); pe_track(i, field) for i < n, field 0..=6
//! ```
//!
//! Track fields: 0 id, 1 cx, 2 cy, 3 w, 4 h (pixels), 5 state (0 tentative,
//! 1 confirmed, 2 lost), 6 updated in this frame (0 or 1).

use std::cell::RefCell;

use nalgebra::{Matrix4, Vector4};
use perception_core::{
    BBox2D, BBox2DObservation, BBox2DTracker, BBox2DTrackerConfig, ClassLabel, FrameId,
    ObservationBatch2D, SensorId, Timestamp, TrackLifecycleState, TrackerConfig,
};

/// Number of values per track returned by [`CameraTracker::end`].
pub const TRACK_FIELDS: usize = 7;

/// Person tracker for one video stream.
pub struct CameraTracker {
    tracker: BBox2DTracker,
    sequence: u64,
    last_ns: Option<i64>,
    pending_ns: i64,
    pending: Vec<BBox2DObservation>,
    tracks: Vec<[f64; TRACK_FIELDS]>,
}

impl Default for CameraTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraTracker {
    /// Creates a tracker with settings for people in phone video (2-30 fps).
    pub fn new() -> Self {
        let config = BBox2DTrackerConfig {
            common: TrackerConfig {
                // Times, not frame counts, so slow phones (a few fps) still
                // confirm tracks.
                confirm_min_observation_times: 3,
                tentative_timeout_ns: 1_500_000_000,
                confirmed_to_lost_ns: 1_000_000_000,
                lost_to_delete_ns: 2_000_000_000,
                new_track_score_threshold: Some(0.4),
                ..TrackerConfig::default()
            },
            frame_id: FrameId::new("camera"),
            gate_min_iou: 0.2,
            ..BBox2DTrackerConfig::default()
        };
        CameraTracker {
            tracker: BBox2DTracker::new(config).expect("valid configuration"),
            sequence: 0,
            last_ns: None,
            pending_ns: 0,
            pending: Vec::new(),
            tracks: Vec::new(),
        }
    }

    /// Starts a frame at `t_ms` milliseconds; detections follow with [`add`].
    ///
    /// [`add`]: CameraTracker::add
    pub fn begin(&mut self, t_ms: f64) {
        let mut ns = (t_ms * 1e6) as i64;
        // The tracker needs strictly increasing times; a repeated or
        // backwards video timestamp is nudged forward by 1 us.
        if let Some(last) = self.last_ns {
            ns = ns.max(last + 1_000);
        }
        self.pending_ns = ns;
        self.pending.clear();
    }

    /// Adds a person detection (centre, size in pixels, score in [0, 1]).
    /// Non-finite or empty boxes are ignored.
    pub fn add(&mut self, cx: f64, cy: f64, w: f64, h: f64, score: f64) {
        if ![cx, cy, w, h, score].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
            return;
        }
        // Position noise grows with the box: about 5 % of its height.
        let s = (0.05 * h).max(1.0);
        let c = (0.1 * h).max(1.0);
        let covariance = Matrix4::from_diagonal(&Vector4::new(s * s, s * s, c * c, c * c));
        self.pending.push(BBox2DObservation {
            bbox: BBox2D::from_center_size(cx, cy, w, h),
            score: score.clamp(0.0, 1.0),
            class_label: Some(ClassLabel::new(1)),
            covariance,
        });
    }

    /// Runs the tracker on the frame's detections; returns the tracks.
    /// On an internal error the tracker starts over.
    pub fn end(&mut self) -> &[[f64; TRACK_FIELDS]] {
        let batch = ObservationBatch2D {
            sensor_id: SensorId::new(1),
            sequence: self.sequence,
            timestamp: Timestamp::from_nanos(self.pending_ns),
            frame_id: FrameId::new("camera"),
            observations: std::mem::take(&mut self.pending),
        };
        self.sequence += 1;
        self.last_ns = Some(self.pending_ns);
        self.tracks.clear();
        match self.tracker.process_batch(&batch) {
            Ok(snapshots) => {
                for s in snapshots {
                    let state = match s.lifecycle {
                        TrackLifecycleState::Tentative => 0.0,
                        TrackLifecycleState::Confirmed => 1.0,
                        TrackLifecycleState::Lost => 2.0,
                        TrackLifecycleState::Deleted => continue,
                    };
                    let centre = s.bbox.center();
                    self.tracks.push([
                        s.track_id.value() as f64,
                        centre.x,
                        centre.y,
                        s.bbox.width(),
                        s.bbox.height(),
                        state,
                        f64::from(u8::from(s.updated_in_last_batch)),
                    ]);
                }
            }
            Err(_) => *self = CameraTracker::new(),
        }
        &self.tracks
    }
}

thread_local! {
    static TRACKER: RefCell<CameraTracker> = RefCell::new(CameraTracker::new());
}

/// Forgets every track.
#[no_mangle]
pub extern "C" fn pe_reset() {
    TRACKER.with(|t| *t.borrow_mut() = CameraTracker::new());
}

/// Starts a frame at `t_ms` milliseconds.
#[no_mangle]
pub extern "C" fn pe_begin(t_ms: f64) {
    TRACKER.with(|t| t.borrow_mut().begin(t_ms));
}

/// Adds a detection to the current frame.
#[no_mangle]
pub extern "C" fn pe_add(cx: f64, cy: f64, w: f64, h: f64, score: f64) {
    TRACKER.with(|t| t.borrow_mut().add(cx, cy, w, h, score));
}

/// Tracks the current frame; returns the number of tracks.
#[no_mangle]
pub extern "C" fn pe_end() -> u32 {
    TRACKER.with(|t| t.borrow_mut().end().len() as u32)
}

/// Field `field` (see the crate docs) of track `index` from the last
/// [`pe_end`]; NaN when out of range.
#[no_mangle]
pub extern "C" fn pe_track(index: u32, field: u32) -> f64 {
    TRACKER.with(|t| {
        t.borrow()
            .tracks
            .get(index as usize)
            .and_then(|track| track.get(field as usize))
            .copied()
            .unwrap_or(f64::NAN)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walker_keeps_one_confirmed_id() {
        let mut tracker = CameraTracker::new();
        let mut ids = Vec::new();
        for frame in 0..30 {
            let t = frame as f64 * 50.0;
            tracker.begin(t);
            tracker.add(100.0 + frame as f64 * 4.0, 200.0, 60.0, 160.0, 0.9);
            tracker.add(500.0, 210.0, 50.0, 150.0, 0.8);
            let tracks = tracker.end().to_vec();
            if frame >= 5 {
                assert_eq!(tracks.iter().filter(|t| t[5] == 1.0).count(), 2);
                ids.push(tracks.iter().map(|t| t[0] as u64).collect::<Vec<_>>());
            }
        }
        assert!(ids.windows(2).all(|w| w[0] == w[1]), "ids changed: {ids:?}");
    }

    #[test]
    fn repeated_timestamps_and_bad_boxes_are_tolerated() {
        let mut tracker = CameraTracker::new();
        for _ in 0..5 {
            tracker.begin(1000.0);
            tracker.add(f64::NAN, 0.0, 10.0, 10.0, 0.9);
            tracker.add(50.0, 50.0, 0.0, 10.0, 0.9);
            tracker.add(50.0, 50.0, 20.0, 40.0, 0.9);
            assert!(tracker.end().len() <= 1);
        }
    }

    #[test]
    fn exports_report_tracks() {
        pe_reset();
        for frame in 0..4 {
            pe_begin(frame as f64 * 33.0);
            pe_add(320.0, 240.0, 80.0, 200.0, 0.9);
            assert_eq!(pe_end(), 1);
        }
        assert_eq!(pe_track(0, 5), 1.0);
        assert!(pe_track(5, 0).is_nan());
    }
}
