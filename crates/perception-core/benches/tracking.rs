//! Benchmarks for the tracking core.
//!
//! These measure **only** the tracking core: observation batches are prepared
//! off the clock and no detector, video decoding or other sensor processing is
//! included. The numbers are therefore not an end-to-end FPS figure.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p perception-core
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::{Matrix3, Matrix4, Vector3};
use perception_core::{
    BBox2D, BBox2DObservation, BBox2DTracker, BBox2DTrackerConfig, Cartesian3DTracker,
    Cartesian3DTrackerConfig, FrameId, ObservationBatch2D, ObservationBatch3D,
    Position3DObservation, SensorId, Timestamp, TrackerConfig,
};

const FRAMES: usize = 20;
const PERIOD_NS: i64 = 33_333_333;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn common_config() -> TrackerConfig {
    TrackerConfig {
        confirm_min_observation_times: 1,
        confirmed_to_lost_ns: 10_000_000_000,
        lost_to_delete_ns: 20_000_000_000,
        tentative_timeout_ns: 10_000_000_000,
        new_track_score_threshold: Some(0.3),
        max_contributing_sensors: 8,
    }
}

fn make_batches_2d(objects: usize) -> Vec<ObservationBatch2D> {
    let columns = (objects as f64).sqrt().ceil() as usize;
    let spacing = 200.0;
    let mut rng = SplitMix64(0xC0FF_EE00);

    (0..FRAMES)
        .map(|frame| {
            let t = frame as i64 * PERIOD_NS;
            let observations = (0..objects)
                .map(|i| {
                    let col = i % columns;
                    let row = i / columns;
                    let x = col as f64 * spacing + 5.0 * frame as f64 + (rng.next_f64() - 0.5);
                    let y = row as f64 * spacing + 2.0 * frame as f64 + (rng.next_f64() - 0.5);
                    BBox2DObservation {
                        bbox: BBox2D::from_center_size(x, y, 40.0, 40.0),
                        score: 0.9,
                        class_label: None,
                        covariance: Matrix4::<f64>::identity() * 0.25,
                    }
                })
                .collect();
            ObservationBatch2D {
                sensor_id: SensorId::new(1),
                sequence: frame as u64,
                timestamp: Timestamp::from_nanos(t),
                frame_id: FrameId::new("camera"),
                observations,
            }
        })
        .collect()
}

fn make_batches_3d(objects: usize) -> Vec<ObservationBatch3D> {
    let columns = (objects as f64).sqrt().ceil() as usize;
    let spacing = 20.0;
    let mut rng = SplitMix64(0xDEAD_BEEF);

    (0..FRAMES)
        .map(|frame| {
            let t = frame as i64 * PERIOD_NS;
            let observations = (0..objects)
                .map(|i| {
                    let col = i % columns;
                    let row = i / columns;
                    let x = col as f64 * spacing + 1.0 * frame as f64 + (rng.next_f64() - 0.5);
                    let y = row as f64 * spacing + 0.5 * frame as f64 + (rng.next_f64() - 0.5);
                    Position3DObservation {
                        position: Vector3::new(x, y, 0.0),
                        covariance: Matrix3::<f64>::identity() * 0.25,
                        class_label: None,
                        appearance: None,
                    }
                })
                .collect();
            ObservationBatch3D {
                sensor_id: SensorId::new(1),
                sequence: frame as u64,
                timestamp: Timestamp::from_nanos(t),
                frame_id: FrameId::new("world"),
                observations,
            }
        })
        .collect()
}

fn bench_track_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("track_2d");
    for objects in [10usize, 100, 1000] {
        let batches = make_batches_2d(objects);
        let config = BBox2DTrackerConfig {
            common: common_config(),
            gate_min_iou: 0.1,
            ..BBox2DTrackerConfig::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(objects), &objects, |b, _| {
            b.iter(|| {
                let mut tracker = BBox2DTracker::new(config.clone()).unwrap();
                for batch in &batches {
                    tracker.process_batch(batch).unwrap();
                }
                black_box(tracker.active_track_count())
            });
        });
    }
    group.finish();
}

fn bench_track_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("track_3d_fusion");
    for objects in [10usize, 100, 1000] {
        let batches = make_batches_3d(objects);
        let config = Cartesian3DTrackerConfig {
            common: common_config(),
            gate_squared_mahalanobis: 100.0,
            ..Cartesian3DTrackerConfig::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(objects), &objects, |b, _| {
            b.iter(|| {
                let mut tracker = Cartesian3DTracker::new(config.clone()).unwrap();
                for batch in &batches {
                    tracker.process_batch(batch).unwrap();
                }
                black_box(tracker.active_track_count())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_track_2d, bench_track_3d);
criterion_main!(benches);
