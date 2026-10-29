//! Deterministic synthetic scenarios used by the demos and CLI tests.
//!
//! The data is entirely synthetic and is **not** validated against any real
//! sensor. Ground truth is produced alongside the observations but is never
//! passed to a tracker; it exists only for evaluation and visualization.
//!
//! Seeding is explicit and the PRNG is a small splitmix64 implementation, so
//! every run with the same seed produces byte-identical output.

use factor_tracker_core::{
    BBox2D, BBox2DObservation, FrameId, ObservationBatch2D, ObservationBatch3D,
    Position3DObservation, RigidTransform3D, SensorId, Timestamp,
};
use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::jsonl::{
    CliResult, GroundTruthRecord, Truth2dObject as RecordTruth2dObject,
    Truth3dObject as RecordTruth3dObject, SCHEMA_VERSION,
};

/// Deterministic splitmix64 PRNG.
pub struct Rng(u64);

impl Rng {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// Next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    /// Bernoulli trial with success probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

/// Ground-truth 2D object.
#[derive(Debug, Clone, PartialEq)]
pub struct Truth2dObject {
    /// Ground-truth object id.
    pub truth_id: u32,
    /// Class id.
    pub class: u32,
    /// Box in pixels.
    pub bbox: BBox2D,
    /// The object existed at this time.
    pub present: bool,
    /// The synthetic sensor detected it at this time.
    pub detected: bool,
}

/// Ground-truth 2D frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Truth2dFrame {
    /// Timestamp.
    pub timestamp: Timestamp,
    /// Objects alive at this time.
    pub objects: Vec<Truth2dObject>,
}

/// 2D synthetic scenario.
#[derive(Debug, Clone)]
pub struct Scenario2d {
    /// Observation batches for one camera.
    pub batches: Vec<ObservationBatch2D>,
    /// Ground truth, one entry per frame.
    pub truth: Vec<Truth2dFrame>,
    /// Image width in pixels.
    pub image_width: f64,
    /// Image height in pixels.
    pub image_height: f64,
    /// Sensor ids used.
    pub sensor_ids: Vec<u32>,
}

struct MovingObject2d {
    id: u32,
    class: u32,
    position: [f64; 2],
    velocity: [f64; 2],
    size: [f64; 2],
    start_frame: usize,
    end_frame: usize,
}

/// Generates the 2D demo scenario.
///
/// The scene contains four objects on a 1280x720 image, a temporary full
/// occlusion window, random detector misses and occasional false positives.
/// Objects enter and leave over time and bounce off the image border.
pub fn generate_2d(seed: u64) -> Scenario2d {
    const WIDTH: f64 = 1280.0;
    const HEIGHT: f64 = 720.0;
    const FRAMES: usize = 240;
    const PERIOD_NS: i64 = 33_333_333;

    let mut rng = Rng::new(seed);
    let mut objects = vec![
        MovingObject2d {
            id: 1,
            class: 1,
            position: [120.0, 180.0],
            velocity: [90.0, 0.0],
            size: [120.0, 70.0],
            start_frame: 0,
            end_frame: FRAMES,
        },
        MovingObject2d {
            id: 2,
            class: 2,
            position: [1050.0, 600.0],
            velocity: [-70.0, -40.0],
            size: [45.0, 90.0],
            start_frame: 10,
            end_frame: 220,
        },
        MovingObject2d {
            id: 3,
            class: 1,
            position: [60.0, 620.0],
            velocity: [110.0, -50.0],
            size: [130.0, 75.0],
            start_frame: 60,
            end_frame: FRAMES,
        },
        MovingObject2d {
            id: 4,
            class: 2,
            position: [640.0, 40.0],
            velocity: [30.0, 100.0],
            size: [50.0, 95.0],
            start_frame: 90,
            end_frame: 200,
        },
    ];

    let dt = PERIOD_NS as f64 / 1.0e9;
    let mut batches = Vec::with_capacity(FRAMES);
    let mut truth = Vec::with_capacity(FRAMES);

    for frame in 0..FRAMES {
        let mut truth_objects = Vec::new();
        let mut observations = Vec::new();

        for object in &mut objects {
            if frame < object.start_frame || frame >= object.end_frame {
                continue;
            }
            // Advance and bounce inside the image.
            for axis in 0..2 {
                object.position[axis] += object.velocity[axis] * dt;
            }
            for (axis, limit) in [(0usize, WIDTH), (1usize, HEIGHT)] {
                let half = object.size[axis] / 2.0;
                if object.position[axis] < half {
                    object.position[axis] = half;
                    object.velocity[axis] = object.velocity[axis].abs();
                } else if object.position[axis] > limit - half {
                    object.position[axis] = limit - half;
                    object.velocity[axis] = -object.velocity[axis].abs();
                }
            }

            let bbox = BBox2D::from_center_size(
                object.position[0],
                object.position[1],
                object.size[0],
                object.size[1],
            );

            // Object 3 is fully occluded for a window of frames.
            let occluded = object.id == 3 && (120..150).contains(&frame);
            let detected = !occluded && !rng.chance(0.06);
            if detected {
                let cx = object.position[0] + 1.5 * rng.normal();
                let cy = object.position[1] + 1.5 * rng.normal();
                let w = (object.size[0] + 1.5 * rng.normal()).max(2.0);
                let h = (object.size[1] + 1.5 * rng.normal()).max(2.0);
                observations.push(BBox2DObservation {
                    bbox: BBox2D::from_center_size(cx, cy, w, h),
                    score: 0.85 + 0.1 * rng.next_f64(),
                    class_label: Some(factor_tracker_core::ClassLabel::new(object.class)),
                    covariance: Matrix4::<f64>::identity() * 4.0,
                });
            }

            truth_objects.push(Truth2dObject {
                truth_id: object.id,
                class: object.class,
                bbox,
                present: true,
                detected,
            });
        }

        // Occasional false positives.
        if rng.chance(0.03) {
            let cx = 40.0 + rng.next_f64() * (WIDTH - 80.0);
            let cy = 40.0 + rng.next_f64() * (HEIGHT - 80.0);
            observations.push(BBox2DObservation {
                bbox: BBox2D::from_center_size(cx, cy, 30.0, 30.0),
                score: 0.5,
                class_label: None,
                covariance: Matrix4::<f64>::identity() * 9.0,
            });
        }

        let timestamp = Timestamp::from_nanos(frame as i64 * PERIOD_NS);
        batches.push(ObservationBatch2D {
            sensor_id: SensorId::new(1),
            sequence: frame as u64,
            timestamp,
            frame_id: FrameId::new("camera"),
            observations,
        });
        truth.push(Truth2dFrame {
            timestamp,
            objects: truth_objects,
        });
    }

    Scenario2d {
        batches,
        truth,
        image_width: WIDTH,
        image_height: HEIGHT,
        sensor_ids: vec![1],
    }
}

/// Ground-truth 3D object.
#[derive(Debug, Clone, PartialEq)]
pub struct Truth3dObject {
    /// Ground-truth object id.
    pub truth_id: u32,
    /// Class id.
    pub class: u32,
    /// Position in meters (world frame).
    pub position: Vector3<f64>,
    /// The object existed at this time.
    pub present: bool,
    /// Whether the primary sensor detected it at this time.
    pub detected: bool,
}

/// Ground-truth 3D frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Truth3dFrame {
    /// Timestamp.
    pub timestamp: Timestamp,
    /// Objects alive at this time.
    pub objects: Vec<Truth3dObject>,
}

/// 3D synthetic multi-sensor scenario.
#[derive(Debug, Clone)]
pub struct Scenario3d {
    /// Observation batches for both sensors, sorted by timestamp.
    pub batches: Vec<ObservationBatch3D>,
    /// Ground truth in the world frame, one entry per frame.
    pub truth: Vec<Truth3dFrame>,
    /// Sensor ids used.
    pub sensor_ids: Vec<u32>,
}

struct MovingObject3d {
    id: u32,
    class: u32,
    kind: Motion3d,
}

enum Motion3d {
    Linear {
        origin: Vector3<f64>,
        velocity: Vector3<f64>,
    },
    Circular {
        center: Vector3<f64>,
        radius: f64,
        rate: f64,
    },
}

impl Motion3d {
    fn position(&self, t: f64) -> Vector3<f64> {
        match self {
            Motion3d::Linear { origin, velocity } => origin + velocity * t,
            Motion3d::Circular {
                center,
                radius,
                rate,
            } => {
                let angle = rate * t;
                center + Vector3::new(radius * angle.cos(), radius * angle.sin(), 0.0)
            }
        }
    }
}

/// Generates the 3D multi-sensor fusion scenario.
///
/// Two sensors observe three objects in a shared `world` frame. Sensor 1 runs
/// at 20 Hz with low noise; sensor 2 runs at 5 Hz with higher noise, is rigidly
/// mounted away from the origin (its observations are transformed with
/// [`RigidTransform3D`]) and pauses for two seconds. There are detector misses
/// and false positives.
pub fn generate_3d(seed: u64) -> Scenario3d {
    const FRAMES: usize = 300;
    const PERIOD_NS: i64 = 50_000_000;

    let mut rng = Rng::new(seed);
    let objects = vec![
        MovingObject3d {
            id: 1,
            class: 1,
            kind: Motion3d::Linear {
                origin: Vector3::new(0.0, 0.0, 1.0),
                velocity: Vector3::new(1.0, 0.2, 0.0),
            },
        },
        MovingObject3d {
            id: 2,
            class: 2,
            kind: Motion3d::Linear {
                origin: Vector3::new(25.0, 12.0, 0.5),
                velocity: Vector3::new(-0.8, 0.5, 0.1),
            },
        },
        MovingObject3d {
            id: 3,
            class: 1,
            kind: Motion3d::Circular {
                center: Vector3::new(12.0, 8.0, 0.0),
                radius: 5.0,
                rate: 0.35,
            },
        },
    ];

    let sensor2_rotation = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let sensor2_translation = Vector3::new(5.0, 0.0, 0.0);
    let sensor2_local_to_world =
        RigidTransform3D::new(sensor2_rotation, sensor2_translation).expect("valid transform");
    // Inverse transform, used to express a world point in sensor 2's frame.
    let sensor2_world_to_local = RigidTransform3D::new(
        sensor2_rotation.transpose(),
        -(sensor2_rotation.transpose() * sensor2_translation),
    )
    .expect("valid transform");

    let mut batches = Vec::new();
    let mut truth = Vec::with_capacity(FRAMES);
    let mut sensor2_sequence = 0u64;

    for frame in 0..FRAMES {
        let timestamp = Timestamp::from_nanos(frame as i64 * PERIOD_NS);
        let t = timestamp.as_nanos() as f64 / 1.0e9;
        let mut truth_objects = Vec::new();
        let mut sensor1_observations = Vec::new();
        let mut sensor2_observations = Vec::new();

        for object in &objects {
            let position = object.kind.position(t);
            let detected = !rng.chance(0.05);
            if detected {
                sensor1_observations.push(Position3DObservation {
                    position: position
                        + Vector3::new(rng.normal(), rng.normal(), rng.normal()) * 0.15,
                    covariance: Matrix3::<f64>::identity() * (0.15 * 0.15),
                    class_label: Some(factor_tracker_core::ClassLabel::new(object.class)),
                    appearance: None,
                    score: None,
                    velocity: None,
                });
            }

            // Sensor 2 runs every fourth frame and pauses inside [6 s, 8 s).
            let sensor2_active = frame % 4 == 0 && !(6.0..8.0).contains(&t);
            if sensor2_active && !rng.chance(0.05) {
                // Express the world point in sensor 2's local frame, add noise
                // there, then map the noisy measurement back to the world with
                // the rigid transform helper.
                let local = sensor2_world_to_local.transform_position(&position);
                let noisy_local =
                    local + Vector3::new(rng.normal(), rng.normal(), rng.normal()) * 0.4;
                let world = sensor2_local_to_world.transform_position(&noisy_local);
                let covariance = sensor2_local_to_world
                    .transform_covariance(&(Matrix3::<f64>::identity() * (0.4 * 0.4)))
                    .expect("valid covariance");
                sensor2_observations.push(Position3DObservation {
                    position: world,
                    covariance,
                    class_label: Some(factor_tracker_core::ClassLabel::new(object.class)),
                    appearance: None,
                    score: None,
                    velocity: None,
                });
            }

            truth_objects.push(Truth3dObject {
                truth_id: object.id,
                class: object.class,
                position,
                present: true,
                detected,
            });
        }

        // False positives.
        if rng.chance(0.02) {
            sensor1_observations.push(Position3DObservation {
                position: Vector3::new(rng.next_f64() * 30.0, rng.next_f64() * 20.0, 0.0),
                covariance: Matrix3::<f64>::identity() * 1.0,
                class_label: None,
                appearance: None,
                score: None,
                velocity: None,
            });
        }

        // Sensor 1 reports every frame, including empty batches.
        batches.push(ObservationBatch3D {
            sensor_id: SensorId::new(1),
            sequence: frame as u64,
            timestamp,
            frame_id: FrameId::new("world"),
            observations: sensor1_observations,
        });
        if frame % 4 == 0 && !(6.0..8.0).contains(&t) {
            batches.push(ObservationBatch3D {
                sensor_id: SensorId::new(2),
                sequence: sensor2_sequence,
                timestamp,
                frame_id: FrameId::new("world"),
                observations: sensor2_observations,
            });
            sensor2_sequence += 1;
        }

        truth.push(Truth3dFrame {
            timestamp,
            objects: truth_objects,
        });
    }

    Scenario3d {
        batches,
        truth,
        sensor_ids: vec![1, 2],
    }
}

impl Scenario2d {
    /// Converts the ground truth into JSONL records.
    pub fn truth_records(&self) -> Vec<GroundTruthRecord> {
        self.truth
            .iter()
            .map(|frame| GroundTruthRecord::Truth2d {
                schema_version: SCHEMA_VERSION,
                timestamp_ns: frame.timestamp.as_nanos(),
                objects: frame
                    .objects
                    .iter()
                    .map(|object| RecordTruth2dObject {
                        truth_id: object.truth_id,
                        class: object.class,
                        bbox: vec![
                            object.bbox.x_min,
                            object.bbox.y_min,
                            object.bbox.x_max,
                            object.bbox.y_max,
                        ],
                        present: object.present,
                        detected: object.detected,
                    })
                    .collect(),
            })
            .collect()
    }
}

impl Scenario3d {
    /// Converts the ground truth into JSONL records.
    pub fn truth_records(&self) -> Vec<GroundTruthRecord> {
        self.truth
            .iter()
            .map(|frame| GroundTruthRecord::Truth3d {
                schema_version: SCHEMA_VERSION,
                timestamp_ns: frame.timestamp.as_nanos(),
                objects: frame
                    .objects
                    .iter()
                    .map(|object| RecordTruth3dObject {
                        truth_id: object.truth_id,
                        class: object.class,
                        position: object.position.iter().copied().collect(),
                        present: object.present,
                        detected: object.detected,
                    })
                    .collect(),
            })
            .collect()
    }
}

/// Parses 2D ground-truth records.
pub fn truth_frames_2d(records: &[GroundTruthRecord]) -> CliResult<Vec<Truth2dFrame>> {
    records
        .iter()
        .map(|record| match record {
            GroundTruthRecord::Truth2d {
                schema_version,
                timestamp_ns,
                objects,
            } => {
                if *schema_version != SCHEMA_VERSION {
                    return Err(format!("unsupported schema_version {schema_version}").into());
                }
                let objects = objects
                    .iter()
                    .map(|object| {
                        if object.bbox.len() != 4 {
                            return Err("truth2d bbox must have 4 values".into());
                        }
                        Ok(Truth2dObject {
                            truth_id: object.truth_id,
                            class: object.class,
                            bbox: BBox2D::from_corners(
                                object.bbox[0],
                                object.bbox[1],
                                object.bbox[2],
                                object.bbox[3],
                            ),
                            present: object.present,
                            detected: object.detected,
                        })
                    })
                    .collect::<CliResult<Vec<_>>>()?;
                Ok(Truth2dFrame {
                    timestamp: Timestamp::from_nanos(*timestamp_ns),
                    objects,
                })
            }
            GroundTruthRecord::Truth3d { .. } => {
                Err("expected truth2d records but found truth3d".into())
            }
        })
        .collect()
}

/// Parses 3D ground-truth records.
pub fn truth_frames_3d(records: &[GroundTruthRecord]) -> CliResult<Vec<Truth3dFrame>> {
    records
        .iter()
        .map(|record| match record {
            GroundTruthRecord::Truth3d {
                schema_version,
                timestamp_ns,
                objects,
            } => {
                if *schema_version != SCHEMA_VERSION {
                    return Err(format!("unsupported schema_version {schema_version}").into());
                }
                let objects = objects
                    .iter()
                    .map(|object| {
                        if object.position.len() != 3 {
                            return Err("truth3d position must have 3 values".into());
                        }
                        Ok(Truth3dObject {
                            truth_id: object.truth_id,
                            class: object.class,
                            position: Vector3::new(
                                object.position[0],
                                object.position[1],
                                object.position[2],
                            ),
                            present: object.present,
                            detected: object.detected,
                        })
                    })
                    .collect::<CliResult<Vec<_>>>()?;
                Ok(Truth3dFrame {
                    timestamp: Timestamp::from_nanos(*timestamp_ns),
                    objects,
                })
            }
            GroundTruthRecord::Truth2d { .. } => {
                Err("expected truth3d records but found truth2d".into())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let a = generate_2d(42);
        let b = generate_2d(42);
        assert_eq!(a.batches.len(), b.batches.len());
        assert_eq!(
            a.batches[10].observations.len(),
            b.batches[10].observations.len()
        );
    }

    #[test]
    fn two_d_scenario_has_misses_and_multiple_objects() {
        let scenario = generate_2d(1);
        assert_eq!(scenario.batches.len(), 240);
        let observed = scenario
            .batches
            .iter()
            .map(|batch| batch.observations.len())
            .max()
            .unwrap();
        assert!(observed >= 3, "expected several simultaneous objects");
    }

    #[test]
    fn three_d_scenario_has_two_sensors_and_a_pause() {
        let scenario = generate_3d(1);
        let sensors: std::collections::BTreeSet<u32> = scenario
            .batches
            .iter()
            .map(|batch| batch.sensor_id.value())
            .collect();
        assert_eq!(sensors.len(), 2);
        // The gap corresponds to the 2 second pause of sensor 2.
        let sensor2_times: Vec<i64> = scenario
            .batches
            .iter()
            .filter(|batch| batch.sensor_id == SensorId::new(2))
            .map(|batch| batch.timestamp.as_nanos())
            .collect();
        assert!(sensor2_times.windows(2).any(|w| w[1] - w[0] > 500_000_000));
    }
}
