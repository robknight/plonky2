//! Private executor policy. Consumers configure resources, not kernel tuning.
use super::config::invalid;
use super::WitnessProgramError;

/// Placement of witness values. `Outputs` is the measured general default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WitnessLayout {
    /// Keep the compiler's original canonical ordering.
    Canonical,
    /// Place each operation's outputs together.
    #[default]
    Outputs,
    /// Sort non-root values by copy-constraint representative.
    Representatives,
    /// Interleave outputs in blocks of four operations.
    Block4,
    /// Interleave outputs in blocks of eight operations.
    Block8,
}
impl WitnessLayout {
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Canonical => 0,
            Self::Outputs => 1,
            Self::Representatives => 2,
            Self::Block4 => 3,
            Self::Block8 => 4,
        }
    }
    pub(super) fn from_code(code: u32) -> Result<Self, WitnessProgramError> {
        match code {
            0 => Ok(Self::Canonical),
            1 => Ok(Self::Outputs),
            2 => Ok(Self::Representatives),
            3 => Ok(Self::Block4),
            4 => Ok(Self::Block8),
            _ => Err(invalid("invalid stored witness layout")),
        }
    }
}

/// Execution policy. Automatic execution retains checks for aliases and inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WitnessExecution {
    /// Use validated shared/native kernels, with generic operation fallbacks.
    #[default]
    Auto,
    /// Use the checked interpreter, useful as a control or for diagnosis.
    Checked,
}

/// Dependency scheduling is prepared once and does not change the stored graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WitnessSchedule {
    /// Advance ready chunks independently; use layers on a single-thread pool.
    #[default]
    Auto,
    /// Retain global dependency-layer barriers.
    Layers,
    /// Schedule ready chunks without global barriers (requires `parallel`).
    #[cfg(any(test, feature = "witness-benchmarks"))]
    Dependencies,
}

/// Poseidon selection, independent of the serialized circuit program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WitnessPoseidon {
    /// Batch independent permutations when the compiled backend supports it.
    #[default]
    Auto,
    /// Use individual permutations.
    #[cfg(any(test, feature = "witness-benchmarks"))]
    Scalar,
}

/// Destination for the final witness gather.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WitnessGather {
    /// Use the recommended CPU gather, including mapped GPU staging when applicable.
    #[default]
    Auto,
    /// Request GPU gathering when GPU proving and a validated plan are available.
    #[cfg(any(feature = "gpu", test, feature = "witness-benchmarks"))]
    Gpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    pub layout: WitnessLayout,
    pub chain_length: u8,
    pub execution: WitnessExecution,
    pub schedule: WitnessSchedule,
    pub poseidon: WitnessPoseidon,
    pub gather: WitnessGather,
    #[cfg(test)]
    pub gather_tiles: (usize, usize),
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            layout: WitnessLayout::Outputs,
            chain_length: 4,
            execution: WitnessExecution::Auto,
            schedule: WitnessSchedule::Auto,
            poseidon: WitnessPoseidon::Auto,
            gather: WitnessGather::Auto,
            #[cfg(test)]
            gather_tiles: (16, 256),
        }
    }
}
impl Policy {
    pub(super) fn validate(&self) -> Result<(), WitnessProgramError> {
        #[cfg(all(not(feature = "parallel"), any(test, feature = "witness-benchmarks")))]
        if self.schedule == WitnessSchedule::Dependencies {
            return Err(invalid(
                "dependency scheduling requires the parallel feature",
            ));
        }
        if self.chain_length == 1 || self.chain_length > 8 {
            return Err(invalid("invalid stored chain length"));
        }
        Ok(())
    }
    pub(super) fn compilation_key(&self) -> u32 {
        self.layout.code() | (u32::from(self.chain_length) << 3)
    }
    pub(super) fn with_compilation_key(mut self, key: u32) -> Result<Self, WitnessProgramError> {
        self.layout = WitnessLayout::from_code(key & 7)?;
        self.chain_length =
            u8::try_from(key >> 3).map_err(|_| invalid("invalid stored chain length"))?;
        self.validate()?;
        Ok(self)
    }
    #[cfg(any(test, feature = "witness-benchmarks"))]
    pub(super) fn apply_diagnostics<F: Fn(&str) -> Option<std::ffi::OsString>>(
        &mut self,
        env: &crate::util::environment::Environment<F>,
    ) -> Result<(), WitnessProgramError> {
        if let Some(v) = env.choice(
            "PLONKY2_WITNESS_BENCH_MODE",
            "PLONKY2_WITNESS_BENCH_MODE",
            &[
                ("auto", WitnessExecution::Auto),
                ("checked", WitnessExecution::Checked),
            ],
        )? {
            self.execution = v;
        }
        if let Some(v) = env.choice(
            "PLONKY2_WITNESS_BENCH_SCHEDULE",
            "PLONKY2_WITNESS_BENCH_SCHEDULE",
            &[
                ("auto", WitnessSchedule::Auto),
                ("layers", WitnessSchedule::Layers),
                ("dependencies", WitnessSchedule::Dependencies),
            ],
        )? {
            self.schedule = v;
        }
        if let Some(v) = env.choice(
            "PLONKY2_WITNESS_BENCH_POSEIDON",
            "PLONKY2_WITNESS_BENCH_POSEIDON",
            &[
                ("auto", WitnessPoseidon::Auto),
                ("scalar", WitnessPoseidon::Scalar),
            ],
        )? {
            self.poseidon = v;
        }
        if let Some(v) = env.choice(
            "PLONKY2_WITNESS_BENCH_GATHER",
            "PLONKY2_WITNESS_BENCH_GATHER",
            &[
                ("auto", WitnessGather::Auto),
                ("cpu", WitnessGather::Auto),
                ("gpu", WitnessGather::Gpu),
            ],
        )? {
            self.gather = v;
        }
        self.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_controls_are_separate_from_compilation_policy() {
        let env = crate::util::environment::Environment::with_lookup(|key: &str| match key {
            "PLONKY2_WITNESS_BENCH_MODE" => Some("checked".into()),
            "PLONKY2_WITNESS_BENCH_SCHEDULE" => Some("layers".into()),
            "PLONKY2_WITNESS_BENCH_POSEIDON" => Some("scalar".into()),
            "PLONKY2_WITNESS_BENCH_GATHER" => Some("gpu".into()),
            _ => None,
        });
        let mut p = Policy::default();
        p.apply_diagnostics(&env).unwrap();
        assert_eq!(p.compilation_key(), Policy::default().compilation_key());
        assert_eq!(p.execution, WitnessExecution::Checked);
        assert_eq!(p.schedule, WitnessSchedule::Layers);
        assert_eq!(p.poseidon, WitnessPoseidon::Scalar);
        assert_eq!(p.gather, WitnessGather::Gpu);
        let env = crate::util::environment::Environment::with_lookup(|key: &str| {
            (key == "PLONKY2_WITNESS_BENCH_MODE").then(|| "bad".into())
        });
        assert!(p
            .apply_diagnostics(&env)
            .unwrap_err()
            .to_string()
            .contains("PLONKY2_WITNESS_BENCH_MODE"));
    }
    #[test]
    fn stored_compilation_policy_is_validated() {
        let p = Policy::default();
        assert_eq!(
            p.clone().with_compilation_key(p.compilation_key()).unwrap(),
            p
        );
        for key in [7, 1 | (1 << 3), 1 | (9 << 3), u32::MAX] {
            assert!(p.clone().with_compilation_key(key).is_err());
        }
    }
}
