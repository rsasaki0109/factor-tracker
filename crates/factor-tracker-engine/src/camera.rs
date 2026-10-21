//! Camera detections, shape metadata and camera-to-track association.
//!
//! Camera association here is **geometric only** (projected-box IoU). Camera
//! class is recorded but never hard-gates a spatial correspondence; the
//! camera's class evidence is fed to the LiDAR association as a soft prior by
//! the engine.

use std::collections::BTreeMap;

use factor_tracker_core::{linear_assignment, BBox2D, ClassLabel, ClassPrior, Timestamp, TrackId};

use crate::error::EngineResult;
use crate::geometry::ProjectedBox;
use crate::input::CameraDetection;

/// Configuration for camera-to-track association.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraAssociationConfig {
    /// Minimum IoU between a projected track box and a camera detection.
    pub gate_min_iou: f64,
}

impl Default for CameraAssociationConfig {
    fn default() -> Self {
        CameraAssociationConfig { gate_min_iou: 0.2 }
    }
}

impl CameraAssociationConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> EngineResult<()> {
        if !self.gate_min_iou.is_finite() || !(0.0..=1.0).contains(&self.gate_min_iou) {
            return Err(crate::error::EngineError::config(
                "gate_min_iou must be within [0, 1]",
            ));
        }
        Ok(())
    }
}

/// Why a camera detection was or was not associated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMatchReason {
    /// Associated with a projected track.
    Accepted,
    /// The best candidate existed but was below the IoU gate.
    GatedIoU,
    /// No projected track was available.
    Unmatched,
}

impl CameraMatchReason {
    /// Stable lowercase name.
    pub const fn as_str(self) -> &'static str {
        match self {
            CameraMatchReason::Accepted => "accepted",
            CameraMatchReason::GatedIoU => "gated_iou",
            CameraMatchReason::Unmatched => "unmatched",
        }
    }
}

/// Association diagnostic for one camera detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraDetectionMatch {
    /// Index within the camera batch.
    pub detection_index: usize,
    /// Detection id.
    pub detection_id: u64,
    /// Associated track, if any.
    pub track_id: Option<TrackId>,
    /// Projected track box used, if any.
    pub projected_bbox: Option<BBox2D>,
    /// Assignment cost (`1 - IoU`), if matched.
    pub cost: Option<f64>,
    /// Best IoU seen against any candidate track.
    pub best_iou: f64,
    /// Whether the detection was associated.
    pub accepted: bool,
    /// Reason.
    pub reason: CameraMatchReason,
}

/// Associates camera detections with projected tracks (one-to-one, IoU cost).
///
/// `candidates` contains only tracks that produced a valid projection.
pub fn associate_camera(
    detections: &[CameraDetection],
    candidates: &[(TrackId, ProjectedBox)],
    config: &CameraAssociationConfig,
) -> EngineResult<(Vec<CameraDetectionMatch>, Vec<TrackId>)> {
    let n_tracks = candidates.len();
    let n_detections = detections.len();
    let mut costs = vec![vec![f64::INFINITY; n_detections]; n_tracks];
    let mut best_iou = vec![0.0_f64; n_detections];
    for (i, (_track_id, projected)) in candidates.iter().enumerate() {
        for (j, detection) in detections.iter().enumerate() {
            let iou = projected.bbox.iou(&detection.bbox);
            best_iou[j] = best_iou[j].max(iou);
            if iou >= config.gate_min_iou {
                costs[i][j] = 1.0 - iou;
            }
        }
    }

    let assignment = linear_assignment(&costs, n_detections, 1.0 - config.gate_min_iou, 0.0)?;

    let mut matches: Vec<CameraDetectionMatch> = detections
        .iter()
        .enumerate()
        .map(|(index, detection)| CameraDetectionMatch {
            detection_index: index,
            detection_id: detection.detection_id,
            track_id: None,
            projected_bbox: None,
            cost: None,
            best_iou: best_iou[index],
            accepted: false,
            reason: if n_tracks == 0 {
                CameraMatchReason::Unmatched
            } else {
                CameraMatchReason::GatedIoU
            },
        })
        .collect();

    for &(i, j) in &assignment.pairs {
        matches[j].track_id = Some(candidates[i].0);
        matches[j].projected_bbox = Some(candidates[i].1.bbox);
        matches[j].cost = Some(costs[i][j]);
        matches[j].accepted = true;
        matches[j].reason = CameraMatchReason::Accepted;
    }

    let unmatched_tracks = assignment
        .unmatched_rows
        .iter()
        .map(|&i| candidates[i].0)
        .collect();

    Ok((matches, unmatched_tracks))
}

/// Configuration for camera class evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassEvidenceConfig {
    /// Half-life for exponential decay of accumulated class evidence.
    pub half_life_ns: i64,
    /// Minimum number of accepted camera matches before evidence is usable.
    pub min_hits: u32,
    /// Minimum normalized evidence ratio in `[0, 1]` to emit a prior.
    pub min_confidence: f64,
    /// Evidence older than this is ignored.
    pub max_age_ns: i64,
}

impl Default for ClassEvidenceConfig {
    fn default() -> Self {
        ClassEvidenceConfig {
            half_life_ns: 2_000_000_000,
            min_hits: 3,
            min_confidence: 0.6,
            max_age_ns: 1_500_000_000,
        }
    }
}

impl ClassEvidenceConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> EngineResult<()> {
        if self.half_life_ns <= 0 || self.max_age_ns <= 0 {
            return Err(crate::error::EngineError::config(
                "class evidence half_life_ns and max_age_ns must be > 0",
            ));
        }
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(crate::error::EngineError::config(
                "min_confidence must be within [0, 1]",
            ));
        }
        Ok(())
    }
}

/// Accumulated, exponentially-decaying class evidence for one track.
///
/// `confidence` emitted by [`ClassEvidence::best`] is a normalized ratio of
/// accumulated detector scores. It is **not** a calibrated probability.
#[derive(Debug, Clone)]
pub struct ClassEvidence {
    scores: BTreeMap<u32, f64>,
    last_update: Timestamp,
    hits: u32,
}

impl ClassEvidence {
    /// Creates empty evidence anchored at `timestamp`.
    pub fn new(timestamp: Timestamp) -> Self {
        ClassEvidence {
            scores: BTreeMap::new(),
            last_update: timestamp,
            hits: 0,
        }
    }

    /// Adds a class observation with the given detector score.
    pub fn add(&mut self, label: ClassLabel, score: f64, timestamp: Timestamp, half_life_ns: i64) {
        self.decay_to(timestamp, half_life_ns);
        *self.scores.entry(label.value()).or_insert(0.0) += score.clamp(0.0, 1.0).max(0.0);
        self.hits = self.hits.saturating_add(1);
        self.last_update = timestamp;
    }

    /// Last time evidence was updated.
    pub fn last_update(&self) -> Timestamp {
        self.last_update
    }

    /// Number of accepted camera matches contributing evidence.
    pub fn hits(&self) -> u32 {
        self.hits
    }

    fn decay_to(&mut self, timestamp: Timestamp, half_life_ns: i64) {
        let elapsed = timestamp
            .checked_duration_since(self.last_update)
            .ok()
            .filter(|ns| *ns > 0)
            .unwrap_or(0);
        if elapsed == 0 {
            return;
        }
        let factor = 0.5_f64.powf(elapsed as f64 / half_life_ns as f64);
        for score in self.scores.values_mut() {
            *score *= factor;
        }
    }

    /// Returns the current soft class prior, if the evidence is dense enough and
    /// fresh enough.
    pub fn best(&self, now: Timestamp, config: &ClassEvidenceConfig) -> Option<ClassPrior> {
        if self.hits < config.min_hits {
            return None;
        }
        let age = now
            .checked_duration_since(self.last_update)
            .ok()
            .filter(|ns| *ns >= 0)
            .unwrap_or(0);
        if age > config.max_age_ns {
            return None;
        }
        let mut scores = self.scores.clone();
        let elapsed = now
            .checked_duration_since(self.last_update)
            .ok()
            .filter(|ns| *ns > 0)
            .unwrap_or(0);
        if elapsed > 0 {
            let factor = 0.5_f64.powf(elapsed as f64 / config.half_life_ns as f64);
            for score in scores.values_mut() {
                *score *= factor;
            }
        }
        let total: f64 = scores.values().sum();
        if total <= 0.0 {
            return None;
        }
        let (best_label, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))?;
        let confidence = best_score / total;
        if confidence < config.min_confidence {
            return None;
        }
        Some(ClassPrior {
            label: ClassLabel::new(*best_label),
            confidence,
        })
    }
}
