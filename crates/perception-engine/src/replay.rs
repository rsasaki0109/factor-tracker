//! Convenience runner that feeds an input recording through the engine.

use perception_core::Timestamp;

use crate::engine::{CameraAssistedLidarEngine, EngineConfig};
use crate::error::EngineResult;
use crate::geometry::{Calibration, PoseTrack};
use crate::input::EngineBatch;

/// Runs a full replay: ingest each batch at its arrival time, poll, then flush.
pub fn run(
    calibration: Calibration,
    poses: Option<PoseTrack>,
    config: EngineConfig,
    inputs: &[(EngineBatch, Timestamp)],
) -> EngineResult<CameraAssistedLidarEngine> {
    let mut engine = CameraAssistedLidarEngine::new(config, calibration, poses)?;
    for (batch, arrival) in inputs {
        engine.ingest(batch.clone(), *arrival);
        engine.poll(*arrival)?;
    }
    engine.flush()?;
    Ok(engine)
}
