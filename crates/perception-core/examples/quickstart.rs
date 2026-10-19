//! Minimal end-to-end example for both trackers.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p perception-core --example quickstart
//! ```

use nalgebra::{Matrix3, Matrix4, Vector3};
use perception_core::{
    BBox2D, BBox2DObservation, BBox2DTracker, BBox2DTrackerConfig, Cartesian3DTracker,
    Cartesian3DTrackerConfig, FrameId, ObservationBatch2D, ObservationBatch3D,
    Position3DObservation, SensorId, Timestamp,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 2D bounding boxes on one camera ---------------------------------
    let mut tracker_2d = BBox2DTracker::new(BBox2DTrackerConfig::default())?;
    for step in 0..3u64 {
        let measurement = BBox2DObservation {
            bbox: BBox2D::from_center_size(10.0 + 5.0 * step as f64, 20.0, 40.0, 80.0),
            score: 0.95,
            class_label: None,
            covariance: Matrix4::<f64>::identity(),
        };
        let tracks = tracker_2d.process_batch(&ObservationBatch2D {
            sensor_id: SensorId::new(1),
            sequence: step,
            timestamp: Timestamp::from_nanos(step as i64 * 33_000_000),
            frame_id: FrameId::new("camera"),
            observations: vec![measurement],
        })?;
        for track in tracks {
            println!(
                "2D track {} state={} center=({:.1}, {:.1}) vx={:.1}px/s",
                track.track_id,
                track.lifecycle,
                track.bbox.center().x,
                track.bbox.center().y,
                track.velocity.x,
            );
        }
    }

    // ---- 3D positions fused from two sensors -----------------------------
    let mut tracker_3d = Cartesian3DTracker::new(Cartesian3DTrackerConfig::default())?;
    let sensors = [(SensorId::new(1), 0.2), (SensorId::new(2), 0.4)];
    for (sequence, (sensor, sigma)) in sensors.into_iter().enumerate() {
        let tracks = tracker_3d.process_batch(&ObservationBatch3D {
            sensor_id: sensor,
            sequence: sequence as u64,
            timestamp: Timestamp::from_nanos(0),
            frame_id: FrameId::new("world"),
            observations: vec![Position3DObservation {
                position: Vector3::new(1.0, 2.0, 3.0),
                covariance: Matrix3::<f64>::identity() * sigma * sigma,
                class_label: None,
                appearance: None,
                score: None,
            }],
        })?;
        for track in tracks {
            println!(
                "3D track {} position=({:.2}, {:.2}, {:.2}) sensors={:?}",
                track.track_id,
                track.position.x,
                track.position.y,
                track.position.z,
                track.contributing_sensors,
            );
        }
    }

    Ok(())
}
