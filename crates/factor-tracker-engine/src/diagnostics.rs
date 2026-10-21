//! Counters and a bounded diagnostic event log.

use factor_tracker_core::{SensorId, Timestamp};
use serde::{Deserialize, Serialize};

use crate::reorder::ReorderCounters;

/// Counters kept by the engine itself (the reorder buffer keeps its own).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineCounters {
    /// LiDAR batches accepted for processing.
    pub lidar_batches: u64,
    /// Camera batches accepted for processing.
    pub camera_batches: u64,
    /// LiDAR detections that could not be transformed (missing pose).
    pub transform_unavailable: u64,
    /// Camera associations rejected by gating.
    pub association_rejected: u64,
    /// Camera matches accepted.
    pub camera_matches: u64,
    /// Camera detections left unmatched.
    pub camera_unmatched: u64,
    /// Batches rejected by frame or content validation.
    pub rejected_invalid: u64,
}

/// All counters, with the reorder buffer's counters embedded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllCounters {
    /// Reorder buffer counters.
    pub reorder: ReorderCounters,
    /// Engine counters.
    pub engine: EngineCounters,
}

/// Severity of a diagnostic event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// Normal informational event.
    Info,
    /// Something was skipped, rejected or dropped.
    Warn,
}

/// A single diagnostic event.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticEvent {
    /// Arrival time at which the event was produced.
    pub arrival_time: Timestamp,
    /// Event time of the batch, if applicable.
    pub event_time: Option<Timestamp>,
    /// Sensor, if applicable.
    pub sensor_id: Option<SensorId>,
    /// Short machine-readable code.
    pub code: &'static str,
    /// Severity.
    pub level: DiagnosticLevel,
    /// Human readable message.
    pub message: String,
}

/// A bounded log of diagnostic events.
#[derive(Debug, Clone)]
pub struct DiagnosticLog {
    events: Vec<DiagnosticEvent>,
    limit: usize,
    dropped: u64,
}

impl DiagnosticLog {
    /// Creates a log that keeps at most `limit` events.
    pub fn new(limit: usize) -> Self {
        DiagnosticLog {
            events: Vec::new(),
            limit: limit.max(1),
            dropped: 0,
        }
    }

    /// Records an event, dropping it (and counting it) when the log is full.
    pub fn push(&mut self, event: DiagnosticEvent) {
        if self.events.len() >= self.limit {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.events.push(event);
    }

    /// Recorded events, oldest first.
    pub fn events(&self) -> &[DiagnosticEvent] {
        &self.events
    }

    /// Number of events dropped because the log was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

impl Default for DiagnosticLog {
    fn default() -> Self {
        DiagnosticLog::new(10_000)
    }
}
