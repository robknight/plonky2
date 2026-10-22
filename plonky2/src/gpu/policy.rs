//! A proof chooses its storage and quotient path before creating commitments.

use std::sync::Arc;

use super::quotient::QuotientProgram;
use super::GpuContext;
use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;

/// Where a proof keeps its commitment leaves and evaluates the quotient.
/// Quotients the GPU cannot evaluate always use host leaves and the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Resident on discrete-memory adapters, hybrid on unified-memory ones.
    #[default]
    Auto,
    /// Leaves stay on the device and the GPU evaluates the whole quotient.
    Resident,
    /// The CPU evaluates a share of the quotient rows. Within a proof on a
    /// unified-memory adapter the leaves stay on the device and the CPU reads
    /// its rows from the mapped buffer; on a discrete adapter, and for
    /// commitments made outside a proof, leaves are kept on the host.
    Hybrid,
}

impl ExecutionMode {
    pub(crate) fn path(self, mappable: bool, supported: bool) -> ExecutionPath {
        if !supported {
            ExecutionPath::CpuQuotient
        } else if self == Self::Resident || (self == Self::Auto && !mappable) {
            ExecutionPath::Resident
        } else {
            ExecutionPath::Hybrid
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionPath {
    Resident,
    Hybrid,
    CpuQuotient,
}

/// The prepared program pins the context while this proof is running. CPU share
/// is a snapshot: another proof's feedback cannot alter this proof's plan.
pub(crate) struct ExecutionPlan {
    pub(crate) path: ExecutionPath,
    pub(crate) prepared: Option<(Arc<GpuContext>, Arc<QuotientProgram>)>,
    pub(crate) cpu_share: f64,
}

impl ExecutionPlan {
    pub(crate) fn new<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        common: &CommonCircuitData<F, D>,
        prover: &ProverOnlyCircuitData<F, C, D>,
    ) -> Self {
        let prepared = super::quotient::prepare(common, prover);
        Self::from_prepared(prepared, None)
    }

    pub(crate) fn from_prepared(
        prepared: Option<(Arc<GpuContext>, Arc<QuotientProgram>)>,
        mode: Option<ExecutionMode>,
    ) -> Self {
        let (path, cpu_share) = match &prepared {
            Some((ctx, _)) => {
                let path = mode.unwrap_or(ctx.options.mode).path(ctx.mappable, true);
                let share = if path == ExecutionPath::Hybrid {
                    *ctx.quotient_cpu_share.lock().unwrap()
                } else {
                    0.0
                };
                (path, share)
            }
            None => (ExecutionPath::CpuQuotient, 0.0),
        };
        log::debug!("gpu: proof plan {path:?}, CPU share {cpu_share:.3}");
        Self {
            path,
            prepared,
            cpu_share,
        }
    }

    /// Whether this proof's commitments keep their leaf rows on the device:
    /// resident, or hybrid where the host maps device buffers directly, so the
    /// CPU quotient share copies only the rows it reads.
    pub(crate) fn retain_leaves(&self) -> bool {
        match self.path {
            ExecutionPath::Resident => true,
            ExecutionPath::Hybrid => self.prepared.as_ref().is_some_and(|(ctx, _)| ctx.mappable),
            ExecutionPath::CpuQuotient => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_preserve_platform_defaults_and_cannot_force_unsupported_quotients() {
        for mode in [
            ExecutionMode::Auto,
            ExecutionMode::Resident,
            ExecutionMode::Hybrid,
        ] {
            for mappable in [false, true] {
                assert_eq!(mode.path(mappable, false), ExecutionPath::CpuQuotient);
                let expected = if mode == ExecutionMode::Resident
                    || (mode == ExecutionMode::Auto && !mappable)
                {
                    ExecutionPath::Resident
                } else {
                    ExecutionPath::Hybrid
                };
                assert_eq!(mode.path(mappable, true), expected);
            }
        }
    }
}
