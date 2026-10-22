//! Time-based track lifecycle management.
//!
//! The lifecycle only looks at elapsed time and at the set of *distinct*
//! observation timestamps. It never counts `process_batch` calls and never
//! looks at how many sensors are involved, so:
//!
//! * two sensors reporting the same timestamp confirm a track only once, and
//! * a sensor that reports an empty batch does not by itself cause another
//!   sensor's track to become `Lost`.

use crate::types::{Timestamp, TrackLifecycleState, TrackerConfig};

/// Lifecycle bookkeeping embedded in every track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrackLifecycle {
    state: TrackLifecycleState,
    created_at: Timestamp,
    last_observation_timestamp: Option<Timestamp>,
    last_counted_timestamp: Option<Timestamp>,
    observation_times_count: u32,
}

impl TrackLifecycle {
    /// Creates a new tentative lifecycle anchored at `created_at`.
    pub(crate) fn new(created_at: Timestamp) -> Self {
        TrackLifecycle {
            state: TrackLifecycleState::Tentative,
            created_at,
            last_observation_timestamp: None,
            last_counted_timestamp: None,
            observation_times_count: 0,
        }
    }

    /// Current lifecycle state.
    pub(crate) fn state(&self) -> TrackLifecycleState {
        self.state
    }

    /// Timestamp of the most recent observation, if any.
    pub(crate) fn last_observation_timestamp(&self) -> Option<Timestamp> {
        self.last_observation_timestamp
    }

    /// Number of distinct observation timestamps seen so far.
    pub(crate) fn observation_times_count(&self) -> u32 {
        self.observation_times_count
    }

    /// Records an observation at `timestamp`, updating confirmation state.
    ///
    /// The distinct-timestamp counter is only incremented once per timestamp,
    /// so same-time multi-sensor observations do not double count.
    pub(crate) fn register_observation(&mut self, timestamp: Timestamp, config: &TrackerConfig) {
        if self.last_counted_timestamp != Some(timestamp) {
            self.observation_times_count = self.observation_times_count.saturating_add(1);
            self.last_counted_timestamp = Some(timestamp);
        }
        self.last_observation_timestamp = Some(timestamp);
        match self.state {
            TrackLifecycleState::Tentative => {
                if self.observation_times_count >= config.confirm_min_observation_times {
                    self.state = TrackLifecycleState::Confirmed;
                }
            }
            TrackLifecycleState::Lost => {
                // Re-acquired: a lost track that is observed again is confirmed.
                self.state = TrackLifecycleState::Confirmed;
            }
            TrackLifecycleState::Confirmed | TrackLifecycleState::Deleted => {}
        }
    }

    /// Counts `count` earlier distinct observation times of the object (for
    /// example sightings made before the track existed) toward confirmation.
    pub(crate) fn credit_observation_times(&mut self, count: u32, config: &TrackerConfig) {
        self.observation_times_count = self.observation_times_count.max(count);
        if self.state == TrackLifecycleState::Tentative
            && self.observation_times_count >= config.confirm_min_observation_times
        {
            self.state = TrackLifecycleState::Confirmed;
        }
    }

    /// Advances time to `now`, applying every time-based transition that has
    /// become due. Returns `true` when the track has reached `Deleted`.
    ///
    /// Transitions are applied repeatedly so that a single large time jump can
    /// carry a track through `Confirmed -> Lost -> Deleted`.
    pub(crate) fn advance(&mut self, now: Timestamp, config: &TrackerConfig) -> bool {
        loop {
            let reference = self.last_observation_timestamp.unwrap_or(self.created_at);
            let age_ns = now
                .checked_duration_since(reference)
                .ok()
                .filter(|age| *age >= 0)
                .unwrap_or(0);
            let previous = self.state;
            match self.state {
                TrackLifecycleState::Tentative => {
                    if age_ns > config.tentative_timeout_ns {
                        // A tentative track that is not confirmed in time is
                        // dropped outright; only confirmed tracks coast as
                        // `Lost`.
                        self.state = TrackLifecycleState::Deleted;
                    }
                }
                TrackLifecycleState::Confirmed => {
                    if age_ns > config.confirmed_to_lost_ns {
                        self.state = TrackLifecycleState::Lost;
                    }
                }
                TrackLifecycleState::Lost => {
                    if age_ns > config.lost_to_delete_ns {
                        self.state = TrackLifecycleState::Deleted;
                    }
                }
                TrackLifecycleState::Deleted => return true,
            }
            if self.state == previous {
                return self.state == TrackLifecycleState::Deleted;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TrackerConfig {
        TrackerConfig {
            confirm_min_observation_times: 2,
            tentative_timeout_ns: 100,
            confirmed_to_lost_ns: 200,
            lost_to_delete_ns: 500,
            new_track_score_threshold: None,
            max_contributing_sensors: 4,
        }
    }

    fn ts(ns: i64) -> Timestamp {
        Timestamp::from_nanos(ns)
    }

    #[test]
    fn confirm_after_distinct_observation_times() {
        let mut life = TrackLifecycle::new(ts(0));
        life.register_observation(ts(10), &config());
        assert_eq!(life.state(), TrackLifecycleState::Tentative);
        life.register_observation(ts(20), &config());
        assert_eq!(life.state(), TrackLifecycleState::Confirmed);
    }

    #[test]
    fn same_timestamp_multiple_sensors_counts_once() {
        let mut life = TrackLifecycle::new(ts(0));
        life.register_observation(ts(10), &config());
        life.register_observation(ts(10), &config());
        life.register_observation(ts(10), &config());
        assert_eq!(life.observation_times_count(), 1);
        assert_eq!(life.state(), TrackLifecycleState::Tentative);
        life.register_observation(ts(11), &config());
        assert_eq!(life.state(), TrackLifecycleState::Confirmed);
    }

    #[test]
    fn tentative_timeout_deletes() {
        let mut life = TrackLifecycle::new(ts(0));
        assert!(!life.advance(ts(100), &config()));
        assert_eq!(life.state(), TrackLifecycleState::Tentative);
        assert!(life.advance(ts(101), &config()));
        assert_eq!(life.state(), TrackLifecycleState::Deleted);
    }

    #[test]
    fn confirmed_to_lost_to_deleted_over_large_jump() {
        let mut life = TrackLifecycle::new(ts(0));
        life.register_observation(ts(0), &config());
        life.register_observation(ts(1), &config());
        assert_eq!(life.state(), TrackLifecycleState::Confirmed);

        // A single huge jump skips through Lost to Deleted.
        assert!(life.advance(ts(10_000), &config()));
        assert_eq!(life.state(), TrackLifecycleState::Deleted);
    }

    #[test]
    fn lost_track_is_regained_on_observation() {
        let mut life = TrackLifecycle::new(ts(0));
        life.register_observation(ts(0), &config());
        life.register_observation(ts(1), &config());
        assert!(!life.advance(ts(202), &config()));
        assert_eq!(life.state(), TrackLifecycleState::Lost);
        life.register_observation(ts(300), &config());
        assert_eq!(life.state(), TrackLifecycleState::Confirmed);
    }

    #[test]
    fn lost_track_deletes_after_timeout() {
        let mut life = TrackLifecycle::new(ts(0));
        life.register_observation(ts(0), &config());
        life.register_observation(ts(1), &config());
        life.advance(ts(202), &config());
        assert_eq!(life.state(), TrackLifecycleState::Lost);
        assert!(life.advance(ts(600), &config()));
        assert_eq!(life.state(), TrackLifecycleState::Deleted);
    }

    #[test]
    fn confirm_threshold_of_one_confirms_immediately() {
        let mut cfg = config();
        cfg.confirm_min_observation_times = 1;
        let mut life = TrackLifecycle::new(ts(0));
        life.register_observation(ts(0), &cfg);
        assert_eq!(life.state(), TrackLifecycleState::Confirmed);
    }
}
