//! Deterministic injection of latency, reordering, duplication and sensor
//! pauses into an input recording.
//!
//! These are **software fault-injection experiments**. They do not reproduce
//! every real sensor failure mode; they only exercise the engine's asynchronous
//! handling for a fixed, reproducible input.

use factor_tracker_core::Timestamp;

use crate::input::EngineBatch;

/// Which sensor an injection applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorKind {
    /// LiDAR batches.
    Lidar,
    /// Camera batches.
    Camera,
}

impl SensorKind {
    fn matches(self, batch: &EngineBatch) -> bool {
        match self {
            SensorKind::Lidar => batch.is_lidar(),
            SensorKind::Camera => !batch.is_lidar(),
        }
    }
}

/// Injection configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InjectionConfig {
    /// Extra arrival delay applied to LiDAR batches, nanoseconds.
    pub lidar_delay_ns: i64,
    /// Extra arrival delay applied to camera batches, nanoseconds.
    pub camera_delay_ns: i64,
    /// Duplicate every N-th batch (by sequence); `0` disables duplication.
    pub duplicate_every: u64,
    /// Reverse the arrival order inside windows of this size; `<= 1` disables.
    pub reverse_window: usize,
    /// Sensor to pause, if any.
    pub pause_sensor: Option<SensorKind>,
    /// Pause start event time (inclusive).
    pub pause_start_ns: i64,
    /// Pause end event time (inclusive).
    pub pause_end_ns: i64,
}

/// Applies the injection to an input recording.
///
/// Returns a new list in **arrival order** (the order in which the engine
/// should ingest the batches). Event times are never modified.
pub fn apply(
    inputs: &[(EngineBatch, Timestamp)],
    config: &InjectionConfig,
) -> Vec<(EngineBatch, Timestamp)> {
    let mut staged: Vec<(EngineBatch, Timestamp)> = Vec::new();
    for (batch, arrival) in inputs {
        if let Some(paused) = config.pause_sensor {
            if paused.matches(batch) {
                let event = batch.event_time().as_nanos();
                if event >= config.pause_start_ns && event <= config.pause_end_ns {
                    continue;
                }
            }
        }
        let delay = if batch.is_lidar() {
            config.lidar_delay_ns
        } else {
            config.camera_delay_ns
        };
        let new_arrival = Timestamp::from_nanos(arrival.as_nanos().saturating_add(delay));
        staged.push((batch.clone(), new_arrival));
    }

    if config.reverse_window > 1 {
        let mut index = 0;
        while index < staged.len() {
            let end = (index + config.reverse_window).min(staged.len());
            let arrivals: Vec<Timestamp> = staged[index..end]
                .iter()
                .map(|(_, arrival)| *arrival)
                .collect();
            for (offset, arrival) in arrivals.into_iter().rev().enumerate() {
                staged[index + offset].1 = arrival;
            }
            index = end;
        }
    }

    if config.duplicate_every > 0 {
        let mut extra = Vec::new();
        for (batch, arrival) in &staged {
            if batch.sequence() % config.duplicate_every == 0 {
                extra.push((
                    batch.clone(),
                    Timestamp::from_nanos(arrival.as_nanos().saturating_add(1_000)),
                ));
            }
        }
        staged.extend(extra);
    }

    staged.sort_by(|a, b| {
        a.1.as_nanos().cmp(&b.1.as_nanos()).then_with(|| {
            a.0.event_time()
                .as_nanos()
                .cmp(&b.0.event_time().as_nanos())
        })
    });
    staged
}
