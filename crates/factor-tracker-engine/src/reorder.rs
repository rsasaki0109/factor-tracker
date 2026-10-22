//! Bounded, deterministic asynchronous input reordering.
//!
//! Two clocks are kept strictly separate:
//!
//! * **event time** — when the sensor observed the scene. This is what the
//!   tracker uses for state estimation.
//! * **arrival time** — when the engine received the batch. It only influences
//!   buffering decisions (hold timeout), never the estimated state.
//!
//! Batches are released in event-time order. Ties are broken deterministically:
//! LiDAR before camera, then by sensor id, then by sequence number. A
//! *watermark* (`max_event_time - reorder_window`) releases in-order batches,
//! and an independent *hold timeout* (`arrival_now - arrival`) prevents
//! starvation when no new events arrive. Neither mechanism is allowed to
//! advance the tracker's internal time.

use std::collections::{BTreeMap, HashSet};

use factor_tracker_core::Timestamp;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, EngineResult};
use crate::input::EngineBatch;

/// Configuration for [`AsyncReorderBuffer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReorderConfig {
    /// Event-time reorder window in nanoseconds.
    pub window_ns: i64,
    /// Maximum time a batch may wait for out-of-order neighbours, measured on
    /// the arrival clock.
    pub max_hold_ns: i64,
    /// Maximum number of buffered batches.
    pub capacity: usize,
}

impl Default for ReorderConfig {
    fn default() -> Self {
        ReorderConfig {
            window_ns: 200_000_000,
            max_hold_ns: 1_000_000_000,
            capacity: 1024,
        }
    }
}

impl ReorderConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> EngineResult<()> {
        if self.window_ns < 0 {
            return Err(EngineError::config("window_ns must be >= 0"));
        }
        if self.max_hold_ns <= 0 {
            return Err(EngineError::config("max_hold_ns must be > 0"));
        }
        if self.capacity == 0 {
            return Err(EngineError::config("capacity must be > 0"));
        }
        Ok(())
    }
}

/// Counters maintained by the reorder buffer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReorderCounters {
    /// Batches accepted into the buffer.
    pub accepted: u64,
    /// Accepted batches that arrived out of event-time order.
    pub reordered: u64,
    /// Batches older than the last released event time.
    pub late: u64,
    /// Retransmissions (same sensor, kind and sequence).
    pub duplicate: u64,
    /// Batches rejected because the buffer was full.
    pub overflow: u64,
    /// Batches released by an explicit [`AsyncReorderBuffer::flush`].
    pub flushed: u64,
    /// Batches released for processing.
    pub released: u64,
}

/// Result of ingesting a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestOutcome {
    /// Buffered (or releasable on the next poll).
    Queued,
    /// Older than the last released event time; dropped.
    Late,
    /// Duplicate of an already-seen `(kind, sensor, sequence)`; dropped.
    Duplicate,
    /// The buffer is full; dropped.
    Overflow,
}

#[derive(Debug)]
struct Pending {
    batch: EngineBatch,
    arrival: Timestamp,
}

/// A bounded reorder buffer for [`EngineBatch`] values.
#[derive(Debug)]
pub struct AsyncReorderBuffer {
    config: ReorderConfig,
    pending: BTreeMap<(i64, u8, u32, u64), Pending>,
    max_event_ns: Option<i64>,
    last_released_ns: Option<i64>,
    seen: HashSet<(u8, u32, u64)>,
    counters: ReorderCounters,
}

fn kind_rank(batch: &EngineBatch) -> u8 {
    // LiDAR is processed before camera at the same event time.
    if batch.is_lidar() {
        0
    } else {
        1
    }
}

impl AsyncReorderBuffer {
    /// Creates a buffer, validating the configuration.
    pub fn new(config: ReorderConfig) -> EngineResult<Self> {
        config.validate()?;
        Ok(AsyncReorderBuffer {
            config,
            pending: BTreeMap::new(),
            max_event_ns: None,
            last_released_ns: None,
            seen: HashSet::new(),
            counters: ReorderCounters::default(),
        })
    }

    /// Configuration.
    pub fn config(&self) -> &ReorderConfig {
        &self.config
    }

    /// Counters.
    pub fn counters(&self) -> &ReorderCounters {
        &self.counters
    }

    /// Number of buffered batches.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Highest event time seen so far.
    pub fn max_event_time(&self) -> Option<Timestamp> {
        self.max_event_ns.map(Timestamp::from_nanos)
    }

    /// Last released event time.
    pub fn last_released_time(&self) -> Option<Timestamp> {
        self.last_released_ns.map(Timestamp::from_nanos)
    }

    /// Ingests a batch. Does not release anything by itself; call
    /// [`AsyncReorderBuffer::poll`] afterwards.
    pub fn ingest(&mut self, batch: EngineBatch, arrival_time: Timestamp) -> IngestOutcome {
        let rank = kind_rank(&batch);
        let sensor = batch.sensor_id().value();
        let sequence = batch.sequence();
        let event_ns = batch.event_time().as_nanos();

        if self.seen.contains(&(rank, sensor, sequence)) {
            self.counters.duplicate = self.counters.duplicate.saturating_add(1);
            return IngestOutcome::Duplicate;
        }
        if let Some(last) = self.last_released_ns {
            if event_ns < last {
                self.counters.late = self.counters.late.saturating_add(1);
                return IngestOutcome::Late;
            }
        }
        if self.pending.len() >= self.config.capacity {
            self.counters.overflow = self.counters.overflow.saturating_add(1);
            return IngestOutcome::Overflow;
        }
        if let Some(max) = self.max_event_ns {
            if event_ns < max {
                self.counters.reordered = self.counters.reordered.saturating_add(1);
            }
        }
        self.max_event_ns = Some(match self.max_event_ns {
            Some(max) => max.max(event_ns),
            None => event_ns,
        });
        self.seen.insert((rank, sensor, sequence));
        self.pending.insert(
            (event_ns, rank, sensor, sequence),
            Pending {
                batch,
                arrival: arrival_time,
            },
        );
        self.counters.accepted = self.counters.accepted.saturating_add(1);
        IngestOutcome::Queued
    }

    /// Releases every batch that is ready at arrival time `arrival_now`,
    /// in event-time order.
    pub fn poll(&mut self, arrival_now: Timestamp) -> Vec<EngineBatch> {
        let watermark = self
            .max_event_ns
            .map(|max| max.saturating_sub(self.config.window_ns));
        let mut released = Vec::new();
        while let Some(&key) = self.pending.keys().next() {
            let pending = self.pending.get(&key).expect("key exists");
            let event_ns = key.0;
            let by_watermark = watermark.is_some_and(|watermark| event_ns <= watermark);
            let arrival_age = arrival_now
                .as_nanos()
                .saturating_sub(pending.arrival.as_nanos());
            let by_hold = arrival_age >= self.config.max_hold_ns;
            if !by_watermark && !by_hold {
                break;
            }
            let released_entry = self.pending.remove(&key).expect("key exists");
            self.last_released_ns = Some(match self.last_released_ns {
                Some(last) => last.max(event_ns),
                None => event_ns,
            });
            self.counters.released = self.counters.released.saturating_add(1);
            released.push(released_entry.batch);
        }
        released
    }

    /// Releases all remaining batches in event-time order.
    pub fn flush(&mut self) -> Vec<EngineBatch> {
        let mut released = Vec::new();
        while let Some(&key) = self.pending.keys().next() {
            let entry = self.pending.remove(&key).expect("key exists");
            self.last_released_ns = Some(match self.last_released_ns {
                Some(last) => last.max(key.0),
                None => key.0,
            });
            self.counters.flushed = self.counters.flushed.saturating_add(1);
            self.counters.released = self.counters.released.saturating_add(1);
            released.push(entry.batch);
        }
        released
    }
}
