//! Wire values the compiled witness executor (`plonk::witness_program`)
//! uploaded through a staging buffer, handed to the prover in place of a host
//! witness matrix.
use std::sync::Arc;

use crate::gpu::{DeviceBuffer, GpuContext};

/// Wire values already on the device in column-major order, handed to the
/// prover in place of a host witness matrix.
pub(crate) struct ResidentWitness {
    pub ctx: Arc<GpuContext>,
    pub wire_values: DeviceBuffer,
}
