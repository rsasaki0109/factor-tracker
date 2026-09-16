//! 2D bounding-box tracking through detection loss and recovery.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p perception-core --example two_d_tracking
//! ```

use nalgebra::Matrix4;
use perception_core::{
    BBox2D, BBox2DObservation, BBox2DTracker, BBox2DTrackerConfig, FrameId, ObservationBatch2D,
    SensorId, Timestamp, TrackerConfig,
};

fn observation(cx: f64) -> BBox2DObservation {
    BBox2DObservation {
        bbox: BBox2D::from_center_size(cx, 100.0, 60.0, 60.0),
        score: 0.9,
        class_label: None,
        covariance: Matrix4::<f64>::identity(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BBox2DTrackerConfig {
        common: TrackerConfig {
            confirm_min_observation_times: 2,
            confirmed_to_lost_ns: 300_000_000,
            lost_to_delete_ns: 1_000_000_000,
            ..TrackerConfig::default()
        },
        ..BBox2DTrackerConfig::default()
    };
    let mut tracker = BBox2DTracker::new(config)?;

    let period_ns = 100_000_000;
    for frame in 0..20i64 {
        let t = Timestamp::from_nanos(frame * period_ns);
        // The object is not detected between frames 6 and 10.
        let detected = !(6..=10).contains(&frame);
        let observations = if detected {
            vec![observation(100.0 + 5.0 * frame as f64)]
        } else {
            vec![]
        };
        let snapshots = tracker.process_batch(&ObservationBatch2D {
            sensor_id: SensorId::new(1),
            sequence: frame as u64,
            timestamp: t,
            frame_id: FrameId::new("camera"),
            observations,
        })?;
        if snapshots.is_empty() {
            println!("frame {frame:2}: no tracks");
        }
        for snapshot in snapshots {
            println!(
                "frame {frame:2}: track {} state={} updated={} prediction_only={}",
                snapshot.track_id,
                snapshot.lifecycle,
                snapshot.updated_in_last_batch,
                snapshot.is_prediction_only(),
            );
        }
    }
    Ok(())
}
