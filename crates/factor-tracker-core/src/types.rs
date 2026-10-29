//! Core data model: identifiers, timestamps, bounding boxes, observations and
//! batches.
//!
//! # Units and conventions
//!
//! * [`Timestamp`] is an integer number of nanoseconds on a single, shared
//!   timeline. All sensors **must** already be synchronized to this timeline
//!   and expressed in the same epoch. The core never estimates clock offsets.
//! * Image-plane quantities ([`BBox2D`], 2D observations) are in **pixels**.
//!   Velocities derived from them are in **pixels per second**.
//! * 3D quantities ([`Position3DObservation`]) are in **meters**, covariance in
//!   **meters squared**, velocities in **meters per second**.
//! * Covariance matrices are expressed for the *measurement vector* of the
//!   relevant observation and follow the column ordering documented on each
//!   type.

use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};

use crate::error::InputError;

/// A point in time, stored as an integer number of nanoseconds.
///
/// The value may be negative (before the epoch). The common timeline is
/// shared by every sensor; the caller is responsible for synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Creates a timestamp from nanoseconds.
    pub const fn from_nanos(nanos: i64) -> Self {
        Timestamp(nanos)
    }

    /// Creates a timestamp from whole seconds.
    pub const fn from_secs(secs: i64) -> Self {
        Timestamp(secs.saturating_mul(1_000_000_000))
    }

    /// Returns the underlying nanosecond value.
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// Returns `self - earlier` in nanoseconds, or [`InputError::TimestampOverflow`]
    /// when the difference is not representable as `i64`.
    pub fn checked_duration_since(self, earlier: Timestamp) -> Result<i64, InputError> {
        self.0
            .checked_sub(earlier.0)
            .ok_or(InputError::TimestampOverflow)
    }
}

impl core::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}ns", self.0)
    }
}

/// Identifies the sensor that produced an observation batch.
///
/// The value is opaque to the core; the caller decides how to map it to a
/// physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorId(u32);

impl SensorId {
    /// Creates a sensor identifier.
    pub const fn new(value: u32) -> Self {
        SensorId(value)
    }

    /// Returns the raw identifier.
    pub const fn value(self) -> u32 {
        self.0
    }
}

impl core::fmt::Display for SensorId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies a track for the lifetime of a tracker instance.
///
/// Identifiers are monotonically increasing and are never reused, even after a
/// track is deleted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackId(u64);

impl TrackId {
    /// Creates a track identifier.
    pub const fn new(value: u64) -> Self {
        TrackId(value)
    }

    /// Returns the raw identifier.
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for TrackId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Names a coordinate frame.
///
/// A [`crate::tracker::BBox2DTracker`] is bound to one camera frame and a
/// [`crate::tracker::Cartesian3DTracker`] to one common 3D tracking frame.
/// Observations whose frame differs are rejected: renaming a frame is never
/// treated as a coordinate transform.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId(String);

impl FrameId {
    /// Creates a frame identifier.
    pub fn new(name: impl Into<String>) -> Self {
        FrameId(name.into())
    }

    /// Returns the frame name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for FrameId {
    fn from(value: &str) -> Self {
        FrameId::new(value)
    }
}

impl From<String> for FrameId {
    fn from(value: String) -> Self {
        FrameId::new(value)
    }
}

impl core::fmt::Display for FrameId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An opaque detector class identifier.
///
/// The core only uses equality when gating associations: two observations with
/// different *known* classes are never associated. `None` means "unknown
/// class" and is treated as compatible with everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassLabel(u32);

impl ClassLabel {
    /// Creates a class label.
    pub const fn new(value: u32) -> Self {
        ClassLabel(value)
    }

    /// Returns the raw class value.
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// A soft class belief supplied by an external (non-positional) source.
///
/// This is used by the engine layer to feed camera-derived class evidence into
/// the LiDAR association as a *soft* cost, never as a hard gate. `confidence`
/// is in `[0, 1]` and only scales the penalty; it is not a calibrated
/// probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassPrior {
    /// The believed class.
    pub label: ClassLabel,
    /// Evidence strength in `[0, 1]`.
    pub confidence: f64,
}

/// An axis-aligned 2D bounding box on an image plane, in pixels.
///
/// Corners are inclusive/exclusive in the usual image convention; only the
/// geometric extents `width = x_max - x_min` and `height = y_max - y_min`
/// matter for tracking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox2D {
    /// Left edge.
    pub x_min: f64,
    /// Top edge.
    pub y_min: f64,
    /// Right edge.
    pub x_max: f64,
    /// Bottom edge.
    pub y_max: f64,
}

impl BBox2D {
    /// Builds a box directly from its corners without validation.
    ///
    /// Prefer [`BBox2D::validate`] before feeding the box to a tracker.
    pub const fn from_corners(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self {
        BBox2D {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// Builds a box from its center and size without validation.
    pub fn from_center_size(cx: f64, cy: f64, width: f64, height: f64) -> Self {
        let half_w = width / 2.0;
        let half_h = height / 2.0;
        BBox2D {
            x_min: cx - half_w,
            y_min: cy - half_h,
            x_max: cx + half_w,
            y_max: cy + half_h,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    /// Height in pixels.
    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    /// Center `[cx, cy]` in pixels.
    pub fn center(&self) -> Vector2<f64> {
        Vector2::new(
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )
    }

    /// Area in square pixels. May be negative for an invalid box.
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Returns `true` when every corner is finite.
    pub fn is_finite(&self) -> bool {
        self.x_min.is_finite()
            && self.y_min.is_finite()
            && self.x_max.is_finite()
            && self.y_max.is_finite()
    }

    /// Returns `true` when width and height are strictly positive and finite.
    pub fn has_positive_extent(&self) -> bool {
        self.is_finite() && self.width() > 0.0 && self.height() > 0.0
    }

    /// Validates that the box is finite and has strictly positive extents.
    pub fn validate(&self) -> Result<(), InputError> {
        if !self.is_finite() {
            return Err(InputError::NonFiniteValue {
                context: "BBox2D corners",
            });
        }
        if self.width() <= 0.0 || self.height() <= 0.0 {
            return Err(InputError::InvalidBBox {
                width: self.width(),
                height: self.height(),
            });
        }
        Ok(())
    }

    /// Intersection-over-union with `other`, in `[0, 1]`.
    ///
    /// Returns `0.0` when either box has non-positive area.
    pub fn iou(&self, other: &BBox2D) -> f64 {
        if !self.has_positive_extent() || !other.has_positive_extent() {
            return 0.0;
        }
        let inter_x_min = self.x_min.max(other.x_min);
        let inter_y_min = self.y_min.max(other.y_min);
        let inter_x_max = self.x_max.min(other.x_max);
        let inter_y_max = self.y_max.min(other.y_max);

        let inter_w = (inter_x_max - inter_x_min).max(0.0);
        let inter_h = (inter_y_max - inter_y_min).max(0.0);
        let intersection = inter_w * inter_h;

        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// A single 2D bounding-box detection.
///
/// The measurement vector is `[cx, cy, w, h]` (center and size, pixels) and
/// [`BBox2DObservation::covariance`] is the associated `4x4` covariance in
/// `pixels^2`, ordered `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BBox2DObservation {
    /// The detected box in pixels.
    pub bbox: BBox2D,
    /// Detector confidence in `[0, 1]`. This is *not* a track existence
    /// probability and is never reported as one.
    pub score: f64,
    /// Optional detector class.
    pub class_label: Option<ClassLabel>,
    /// Measurement covariance for `[cx, cy, w, h]`, in `pixels^2`.
    pub covariance: Matrix4<f64>,
}

impl BBox2DObservation {
    /// Measurement vector `[cx, cy, w, h]`.
    pub fn measurement(&self) -> nalgebra::Vector4<f64> {
        let c = self.bbox.center();
        nalgebra::Vector4::new(c.x, c.y, self.bbox.width(), self.bbox.height())
    }
}

/// A single 3D position detection in a common tracking frame.
///
/// The measurement vector is `[x, y, z]` in meters and
/// [`Position3DObservation::covariance`] is the associated `3x3` covariance in
/// `meters^2`.
#[derive(Debug, Clone, PartialEq)]
pub struct Position3DObservation {
    /// Position `[x, y, z]` in meters.
    pub position: Vector3<f64>,
    /// Position covariance in `meters^2`.
    pub covariance: Matrix3<f64>,
    /// Optional detector class.
    pub class_label: Option<ClassLabel>,
    /// Optional appearance embedding (for example a ReID feature of the
    /// object's image crop). Compared by cosine distance; the vector does not
    /// need to be normalized. Ignored unless
    /// `Cartesian3DTrackerConfig::appearance_weight` is positive.
    pub appearance: Option<Vec<f32>>,
    /// Optional detector confidence in `[0, 1]`. An observation below
    /// `TrackerConfig::new_track_score_threshold` updates existing tracks
    /// but never starts one.
    pub score: Option<f64>,
    /// Optional measured horizontal velocity `[vx, vy]` in m/s, in the
    /// tracking frame (from a detector that regresses velocity, such as
    /// CenterPoint, or from radar). Ignored unless
    /// `Cartesian3DTrackerConfig::velocity_measurement_std` is set.
    pub velocity: Option<Vector2<f64>>,
}

/// A batch of 2D observations from one sensor at one timestamp.
///
/// The batch is the unit of processing. `sequence` is a per-sensor counter
/// that lets the tracker reject retransmissions of a batch it has already
/// fused: the same `(sensor, sequence)` pair must never be processed twice.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationBatch2D {
    /// Sensor that produced the batch.
    pub sensor_id: SensorId,
    /// Per-sensor monotonically increasing sequence number.
    pub sequence: u64,
    /// Observation time on the shared timeline.
    pub timestamp: Timestamp,
    /// Image frame the boxes live in.
    pub frame_id: FrameId,
    /// The detections. May be empty (a "no detection" report).
    pub observations: Vec<BBox2DObservation>,
}

/// A batch of 3D position observations from one sensor at one timestamp.
///
/// Positions must already be expressed in the tracker's common tracking frame;
/// frame mismatches are rejected rather than silently renamed.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationBatch3D {
    /// Sensor that produced the batch.
    pub sensor_id: SensorId,
    /// Per-sensor monotonically increasing sequence number.
    pub sequence: u64,
    /// Observation time on the shared timeline.
    pub timestamp: Timestamp,
    /// Coordinate frame of the positions.
    pub frame_id: FrameId,
    /// The detections. May be empty (a "no detection" report).
    pub observations: Vec<Position3DObservation>,
}

/// Lifecycle state of a track.
///
/// `Deleted` is never emitted in a snapshot: deleted tracks are removed from
/// the tracker. It exists so that transition logic and tests can name the
/// terminal state explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackLifecycleState {
    /// Recently created, not yet confirmed.
    Tentative,
    /// Confirmed and actively maintained.
    Confirmed,
    /// Was confirmed but has not been observed for a while; still predicted.
    Lost,
    /// Terminal state; the track has been removed.
    Deleted,
}

impl TrackLifecycleState {
    /// Stable lowercase name useful for logs and serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            TrackLifecycleState::Tentative => "tentative",
            TrackLifecycleState::Confirmed => "confirmed",
            TrackLifecycleState::Lost => "lost",
            TrackLifecycleState::Deleted => "deleted",
        }
    }
}

impl core::fmt::Display for TrackLifecycleState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Lifecycle parameters shared by both trackers.
///
/// All time-outs are durations in nanoseconds measured from a track's **last
/// observation** (or its creation time for a track that has never been
/// observed). Lifecycle decisions therefore depend only on elapsed time and
/// never on the number of sensors or on how often `process_batch` is called.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackerConfig {
    /// Number of *distinct observation timestamps* required before a tentative
    /// track becomes confirmed. Multiple sensors reporting at the same
    /// timestamp count once.
    pub confirm_min_observation_times: u32,
    /// A tentative track that is not confirmed within this duration of its
    /// last observation is deleted.
    pub tentative_timeout_ns: i64,
    /// A confirmed track with no observation for this long becomes `Lost`.
    pub confirmed_to_lost_ns: i64,
    /// A `Lost` track with no observation for this long (since its last
    /// observation) is deleted.
    pub lost_to_delete_ns: i64,
    /// Optional minimum detection score required to spawn a new track. `None`
    /// spawns tracks from every unmatched observation.
    pub new_track_score_threshold: Option<f64>,
    /// Upper bound on the number of distinct sensor ids recorded per track.
    /// Beyond this, the oldest recorded sensor is dropped.
    pub max_contributing_sensors: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig {
            confirm_min_observation_times: 2,
            tentative_timeout_ns: 1_000_000_000,
            confirmed_to_lost_ns: 500_000_000,
            lost_to_delete_ns: 2_000_000_000,
            new_track_score_threshold: Some(0.3),
            max_contributing_sensors: 16,
        }
    }
}

impl TrackerConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), InputError> {
        if self.tentative_timeout_ns <= 0 {
            return Err(InputError::InvalidConfig {
                reason: "tentative_timeout_ns must be > 0",
            });
        }
        if self.confirmed_to_lost_ns <= 0 {
            return Err(InputError::InvalidConfig {
                reason: "confirmed_to_lost_ns must be > 0",
            });
        }
        if self.lost_to_delete_ns < self.confirmed_to_lost_ns {
            return Err(InputError::InvalidConfig {
                reason: "lost_to_delete_ns must be >= confirmed_to_lost_ns",
            });
        }
        if self.max_contributing_sensors == 0 {
            return Err(InputError::InvalidConfig {
                reason: "max_contributing_sensors must be > 0",
            });
        }
        if let Some(threshold) = self.new_track_score_threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(InputError::InvalidConfig {
                    reason: "new_track_score_threshold must be within [0, 1]",
                });
            }
        }
        Ok(())
    }
}
