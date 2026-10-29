//! 3D multi-sensor fusion with an explicit sensor-to-world transform.
//!
//! Sensor 2 reports in its own local frame; the transform helper converts both
//! the position and the covariance into the shared tracking frame before the
//! observation is submitted.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p factor-tracker-core --example three_d_fusion
//! ```

use factor_tracker_core::{
    Cartesian3DTracker, Cartesian3DTrackerConfig, FrameId, ObservationBatch3D,
    Position3DObservation, RigidTransform3D, SensorId, Timestamp,
};
use nalgebra::{Matrix3, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tracker = Cartesian3DTracker::new(Cartesian3DTrackerConfig::default())?;

    // Sensor 2 is mounted 5 m along +x and rotated 90 degrees about z.
    let rotation = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let translation = Vector3::new(5.0, 0.0, 0.0);
    let sensor_local_to_world = RigidTransform3D::new(rotation, translation)?;
    let sensor_world_to_local =
        RigidTransform3D::new(rotation.transpose(), -(rotation.transpose() * translation))?;

    for step in 0..4i64 {
        let t = Timestamp::from_nanos(step * 100_000_000);

        // Sensor 1 in world coordinates, 0.3 m position standard deviation.
        tracker.process_batch(&ObservationBatch3D {
            sensor_id: SensorId::new(1),
            sequence: step as u64,
            timestamp: t,
            frame_id: FrameId::new("world"),
            observations: vec![Position3DObservation {
                position: Vector3::new(step as f64, 0.0, 1.0),
                covariance: Matrix3::<f64>::identity() * 0.09,
                class_label: None,
                appearance: None,
                score: None,
                velocity: None,
            }],
        })?;

        // The same world point as seen from sensor 2: first express it in the
        // sensor's local frame, then map it (and its covariance) back to the
        // world frame through the rigid transform helper.
        let world_point = Vector3::new(step as f64, 0.0, 1.0);
        let local_position = sensor_world_to_local.transform_position(&world_point);
        let world_position = sensor_local_to_world.transform_position(&local_position);
        let world_covariance =
            sensor_local_to_world.transform_covariance(&(Matrix3::<f64>::identity() * 0.25))?;
        let snapshots = tracker.process_batch(&ObservationBatch3D {
            sensor_id: SensorId::new(2),
            sequence: step as u64,
            timestamp: t,
            frame_id: FrameId::new("world"),
            observations: vec![Position3DObservation {
                position: world_position,
                covariance: world_covariance,
                class_label: None,
                appearance: None,
                score: None,
                velocity: None,
            }],
        })?;

        for snapshot in snapshots {
            println!(
                "t={t} track {} pos=({:.2}, {:.2}, {:.2}) vel=({:.2}, {:.2}, {:.2}) sensors={:?}",
                snapshot.track_id,
                snapshot.position.x,
                snapshot.position.y,
                snapshot.position.z,
                snapshot.velocity.x,
                snapshot.velocity.y,
                snapshot.velocity.z,
                snapshot.contributing_sensors,
            );
        }
    }

    Ok(())
}
