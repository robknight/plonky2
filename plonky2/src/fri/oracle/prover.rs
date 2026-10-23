//! Proof-local commitments can leave coefficients on the device. Public
//! `PolynomialBatch` values keep their existing, fully materialized API.

use super::*;

pub(crate) trait ProverOracle<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>: Sync
{
    /// Commitment metadata and leaves; coefficients may live separately.
    fn batch(&self) -> &PolynomialBatch<F, C, D>;
    fn polynomials(&self) -> &[PolynomialCoeffs<F>];
    #[cfg(feature = "gpu")]
    fn num_polynomials(&self) -> usize;
    #[cfg(feature = "gpu")]
    fn uniform_degree(&self, degree_log: usize) -> bool;
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> ProverOracle<F, C, D>
    for PolynomialBatch<F, C, D>
{
    fn batch(&self) -> &Self {
        self
    }
    fn polynomials(&self) -> &[PolynomialCoeffs<F>] {
        &self.polynomials
    }
    #[cfg(feature = "gpu")]
    fn num_polynomials(&self) -> usize {
        self.polynomials.len()
    }
    #[cfg(feature = "gpu")]
    fn uniform_degree(&self, degree_log: usize) -> bool {
        !self.polynomials.is_empty()
            && self.degree_log == degree_log
            && self.polynomials.iter().all(|p| p.len() == 1 << degree_log)
    }
}

pub(crate) struct ProverPolynomialBatch<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub(crate) batch: PolynomialBatch<F, C, D>,
    #[cfg(feature = "gpu")]
    coefficients: Option<DeviceCoefficients<F>>,
}

#[cfg(feature = "gpu")]
struct DeviceCoefficients<F: RichField> {
    context: std::sync::Arc<crate::gpu::GpuContext>,
    buffer: crate::gpu::DeviceBuffer,
    columns: usize,
    degree: usize,
    host: std::sync::OnceLock<Vec<PolynomialCoeffs<F>>>,
}

#[cfg(feature = "gpu")]
impl<F: RichField> DeviceCoefficients<F> {
    fn read(&self) -> Vec<PolynomialCoeffs<F>> {
        #[cfg(test)]
        crate::gpu::tests::COEFFICIENT_READBACKS.with(|reads| reads.set(reads.get() + 1));
        crate::gpu::lde::audit_coefficient_transfer(
            "download",
            "fallback",
            self.columns,
            self.degree,
            || {
                crate::gpu::lde::read_columns::<F>(
                    &self.context,
                    &self.buffer,
                    self.columns,
                    self.degree,
                )
                .into_iter()
                .map(PolynomialCoeffs::new)
                .collect()
            },
        )
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    From<PolynomialBatch<F, C, D>> for ProverPolynomialBatch<F, C, D>
{
    fn from(batch: PolynomialBatch<F, C, D>) -> Self {
        Self {
            batch,
            #[cfg(feature = "gpu")]
            coefficients: None,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverPolynomialBatch<F, C, D>
{
    #[cfg(feature = "gpu")]
    pub(crate) fn from_device(
        batch: PolynomialBatch<F, C, D>,
        context: std::sync::Arc<crate::gpu::GpuContext>,
        buffer: crate::gpu::DeviceBuffer,
        columns: usize,
        degree: usize,
    ) -> Self {
        debug_assert!(batch.polynomials.is_empty());
        Self {
            batch,
            coefficients: Some(DeviceCoefficients {
                context,
                buffer,
                columns,
                degree,
                host: Default::default(),
            }),
        }
    }

    pub(crate) fn into_host(self) -> PolynomialBatch<F, C, D> {
        #[cfg(feature = "gpu")]
        let mut batch = self.batch;
        #[cfg(not(feature = "gpu"))]
        let batch = self.batch;
        #[cfg(feature = "gpu")]
        if let Some(mut coefficients) = self.coefficients {
            batch.polynomials = coefficients
                .host
                .take()
                .unwrap_or_else(|| coefficients.read());
        }
        batch
    }

    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn host_coefficients_materialized(&self) -> bool {
        self.coefficients
            .as_ref()
            .is_none_or(|c| c.host.get().is_some())
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> ProverOracle<F, C, D>
    for ProverPolynomialBatch<F, C, D>
{
    fn batch(&self) -> &PolynomialBatch<F, C, D> {
        &self.batch
    }
    fn polynomials(&self) -> &[PolynomialCoeffs<F>] {
        #[cfg(feature = "gpu")]
        if let Some(coefficients) = &self.coefficients {
            return coefficients.host.get_or_init(|| coefficients.read());
        }
        &self.batch.polynomials
    }
    #[cfg(feature = "gpu")]
    fn num_polynomials(&self) -> usize {
        #[cfg(feature = "gpu")]
        if let Some(coefficients) = &self.coefficients {
            return coefficients.columns;
        }
        self.batch.polynomials.len()
    }
    #[cfg(feature = "gpu")]
    fn uniform_degree(&self, degree_log: usize) -> bool {
        #[cfg(feature = "gpu")]
        if let Some(coefficients) = &self.coefficients {
            return coefficients.columns > 0
                && self.batch.degree_log == degree_log
                && coefficients.degree == 1 << degree_log;
        }
        self.batch.uniform_degree(degree_log)
    }
}
