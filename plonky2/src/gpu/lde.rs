//! Batched NTTs and the polynomial batch commitment on the device: IFFT of
//! the values, coset LDE, and leaf hashing straight from the column-major
//! evaluations. Resident proofs reuse these columns as their leaves; hybrid
//! proofs write row-major leaves for CPU quotient evaluation. These are read back
//! or retained on the device according to the proof's execution policy. The
//! batch keeps both the coefficients and the extended evaluations on the
//! device for the quotient and the opening phase.

use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::{Field, PrimeField64};
use crate::fri::oracle::{PolynomialBatch, ProverOracle, ProverPolynomialBatch, SALT_SIZE};
use crate::gpu::merkle::{build_tree, supported, LeafSource};
use crate::gpu::policy::LeafStorage;
use crate::gpu::shader::{
    fft_plan, FFT_INVERSE, FFT_LOAD_BITREV, FFT_SHIFT_IN, FFT_SHIFT_OUT, FFT_TO_OUTPUT,
    FFT_WORKGROUP, HASH_WORKGROUP,
};
use crate::gpu::{context, DeviceBuffer, GpuContext};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{MerkleCap, MerkleDigests, MerkleLeaves, MerkleTree};
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::log2_strict;
use crate::util::timing::TimingTree;

/// Profiling-only attribution of coefficient copies; no values are logged.
pub(crate) fn coefficient_audit_enabled() -> bool {
    cfg!(feature = "gpu-profiling") && crate::util::switches::flag("PLONKY2_GPU_COEFFICIENT_AUDIT")
}

pub(crate) fn audit_coefficient_transfer<T>(
    direction: &str,
    stage: &str,
    columns: usize,
    len: usize,
    work: impl FnOnce() -> T,
) -> T {
    let started = coefficient_audit_enabled().then(std::time::Instant::now);
    let result = work();
    if let Some(started) = started {
        eprintln!(
            "GPU_COEFFICIENT_TRANSFER direction={direction} stage={stage} columns={columns} len={len} bytes={} host_ms={:.6}",
            columns * len * 8, started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

/// The device copy of a batch's coefficients (column-major, one polynomial
/// after another): the one its GPU commitment kept, or an upload of its
/// polynomials, which stays cached in the batch within the retained-bytes
/// limit (the circuit's fixed commitment is the batch that gets here).
pub(crate) fn device_coefficients<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &GpuContext,
    batch: &dyn ProverOracle<F, C, D>,
    stage: &str,
) -> DeviceBuffer {
    if let Some(buffer) = batch.batch().device_coeffs.get() {
        if crate::gpu::lde::coefficient_audit_enabled() {
            eprintln!(
                "GPU_COEFFICIENT_CACHE hit=true columns={} len={}",
                batch.num_polynomials(),
                1usize << batch.batch().degree_log
            );
        }
        return buffer;
    }
    let buffer = crate::gpu::lde::audit_coefficient_transfer(
        "upload",
        stage,
        batch.num_polynomials(),
        1usize << batch.batch().degree_log,
        || {
            let columns = batch
                .polynomials()
                .iter()
                .map(|p| p.coeffs.as_slice())
                .collect::<Vec<_>>();
            upload_columns(ctx, "coeffs", &columns)
        },
    );
    batch
        .batch()
        .device_coeffs
        .set_if_within_limit(ctx, buffer.clone());
    buffer
}

/// One batched NTT over `columns` columns of `2^log_n` elements.
pub(crate) struct FftJob<'a> {
    pub log_n: u32,
    pub columns: u32,
    /// Coefficients loaded by the first group: `in_len` per column at stride
    /// `in_stride`, zero beyond `in_len`.
    pub input: &'a wgpu::Buffer,
    pub in_len: u32,
    pub in_stride: u32,
    /// Working buffer of `columns * 2^log_n` elements; holds the result unless
    /// `output` is given.
    pub data: &'a wgpu::Buffer,
    /// Receives the result of the last group when given.
    pub output: Option<&'a wgpu::Buffer>,
    /// Multiply coefficient `k` by `shift^k` on load (coset FFT).
    pub shift_in: Option<GoldilocksField>,
    /// Inverse transform: reverse the output and divide by `2^log_n`.
    pub inverse: bool,
    /// Multiply output `k` by `shift^k` (coset inverse FFT).
    pub shift_out: Option<GoldilocksField>,
}

/// Records the groups of `job` into `encoder`.
pub(crate) fn fft(ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, job: FftJob<'_>) {
    fft_groups(
        ctx,
        encoder,
        job,
        #[cfg(test)]
        None,
    );
}

/// Records the groups of `job` into `encoder`, or only group `only` (for
/// benchmarks; the data buffer's contents are then meaningless).
pub(crate) fn fft_groups(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    job: FftJob<'_>,
    #[cfg(test)] only: Option<usize>,
) {
    assert!(
        job.shift_in.is_none() || job.shift_out.is_none(),
        "a coset shift on both load and store"
    );
    let n = 1u64 << job.log_n;
    let plan = fft_plan(job.log_n);
    let twiddles = ctx.twiddles(job.log_n);
    let shift_pows = match (job.shift_in, job.shift_out) {
        (Some(shift), _) => ctx.shift_pows(shift, job.in_len),
        (_, Some(shift)) => ctx.shift_pows(shift, n as u32),
        _ => ctx.dummy.clone(),
    };
    let n_inv = GoldilocksField::inverse_2exp(job.log_n as usize).to_canonical_u64();
    let output = job.output.unwrap_or(&ctx.dummy_rw);
    let groups = plan.iter().enumerate();
    #[cfg(test)]
    let groups = groups.filter(|(g, _)| only.is_none_or(|o| o == *g));
    let bindings = groups
        .map(|(g, &(s0, stages, low_bits))| {
            let mut flags = 0;
            let mut skip = 0;
            if g == 0 {
                flags |= FFT_LOAD_BITREV;
                if job.shift_in.is_some() {
                    flags |= FFT_SHIFT_IN;
                }
                // The first `log_n - log2(in_len)` stages of a zero-padded
                // transform (contiguous tile, so only as many as the group has)
                // replicate the loaded coefficients; the kernel skips them.
                if job.in_len.is_power_of_two() && u64::from(job.in_len) < n {
                    assert!(s0 == 0 && low_bits == 0);
                    skip = (job.log_n - job.in_len.ilog2()).min(stages);
                }
            }
            if g == plan.len() - 1 {
                if job.inverse {
                    flags |= FFT_INVERSE;
                }
                if job.shift_out.is_some() {
                    flags |= FFT_SHIFT_OUT;
                }
                if job.output.is_some() {
                    flags |= FFT_TO_OUTPUT;
                }
            }
            let params = ctx.uniform(
                "fft params",
                &[
                    job.log_n,
                    s0,
                    stages,
                    low_bits,
                    job.columns,
                    job.in_len,
                    job.in_stride,
                    flags,
                    n_inv as u32,
                    (n_inv >> 32) as u32,
                    skip,
                    0,
                ],
            );
            let workgroups = (n >> (stages + low_bits)) as u32;
            (
                ctx.bind_fft(&params, job.input, job.data, output, &twiddles, &shift_pows),
                workgroups,
            )
        })
        .collect::<Vec<_>>();
    // Named by shape so captures and traces can tell the commitments apart.
    let label = format!(
        "fft {} x 2^{}{}",
        job.columns,
        job.log_n,
        if job.inverse { " inverse" } else { "" }
    );
    let mut pass = ctx.begin_pass(encoder, &label, None);
    pass.set_pipeline(&ctx.fft);
    for (bindings, workgroups) in &bindings {
        pass.set_bind_group(0, bindings, &[]);
        pass.dispatch_workgroups(*workgroups, job.columns, 1);
    }
}

/// The context, if the GPU takes a batch of `num_polys` polynomials of
/// `degree` coefficients extended by `rate_bits`.
pub(crate) fn applicable<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    num_polys: usize,
    degree: usize,
    rate_bits: usize,
    blinding: bool,
) -> Option<std::sync::Arc<GpuContext>> {
    if !supported::<F, C::Hasher>() || num_polys == 0 || !degree.is_power_of_two() {
        return None;
    }
    let log_n = (log2_strict(degree) + rate_bits) as u32;
    let columns = num_polys + if blinding { SALT_SIZE } else { 0 };
    if log_n < crate::gpu::MIN_LOG_LEAVES || columns > u16::MAX as usize {
        return None;
    }
    let ctx = context()?;
    let n = 1u64 << log_n;
    let fits = ctx.fits(columns as u64 * n * 8);
    let workgroups = (n >> FFT_WORKGROUP.trailing_zeros()).max(n.div_ceil(HASH_WORKGROUP as u64))
        <= ctx.max_workgroups as u64;
    (fits && workgroups).then_some(ctx)
}

pub(crate) fn upload_columns<F: PrimeField64>(
    ctx: &GpuContext,
    label: &str,
    columns: &[&[F]],
) -> DeviceBuffer {
    upload_columns_with_timing(ctx, label, columns, None)
}

fn upload_columns_with_timing<F: PrimeField64>(
    ctx: &GpuContext,
    label: &str,
    columns: &[&[F]],
    mut timing: Option<&mut TimingTree>,
) -> DeviceBuffer {
    if !columns.is_empty() {
        let column_bytes = columns[0].len() * 8;
        if column_bytes > 0 {
            let buffer =
                super::timed_optional(timing.as_deref_mut(), "gpu: allocate mapped upload", || {
                    ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (column_bytes * columns.len()).max(16).next_multiple_of(16) as u64,
                        usage: ctx.storage_usage(),
                        mapped_at_creation: true,
                    })
                });
            super::timed_optional(
                timing.as_deref_mut(),
                "gpu: pack and copy upload columns",
                || {
                    columns.par_iter().enumerate().for_each(|(i, column)| {
                        assert_eq!(column.len() * 8, column_bytes);
                        let start = (i * column_bytes) as u64;
                        let mut view = buffer
                            .slice(start..start + column_bytes as u64)
                            .get_mapped_range_mut()
                            .expect("mapped upload");
                        let (words, _) = view.slice(..).into_chunks::<8>();
                        for (word, value) in words.into_iter().zip(*column) {
                            word.write(value.to_noncanonical_u64().to_le_bytes());
                        }
                    })
                },
            );
            super::timed_optional(timing, "gpu: unmap upload", || buffer.unmap());
            return DeviceBuffer::unpooled(buffer);
        }
    }
    let words = super::timed_optional(timing.as_deref_mut(), "gpu: pack upload columns", || {
        columns
            .par_iter()
            .flat_map_iter(|column| column.iter().map(|x| x.to_noncanonical_u64()))
            .collect::<Vec<_>>()
    });
    super::timed_optional(timing, "gpu: allocate and copy packed upload", || {
        ctx.upload(label, &words)
    })
}

/// Salt field elements per independently seeded chunk.
const SALT_CHUNK: usize = 1 << 15;

/// `columns` salt columns of `n` field elements as upload words, column-major:
/// sampled in parallel chunks, each from its own OS-seeded ChaCha20 generator
/// (dropped, and so zeroized, with its chunk), so all cores take part rather
/// than one per column. Drawing every element from the OS (`F::rand_vec`)
/// would cost about as much as the whole LDE at these sizes.
fn salt_words<F: PrimeField64>(columns: usize, n: usize) -> Vec<u64> {
    let mut words = vec![0u64; columns * n];
    words.par_chunks_mut(SALT_CHUNK).for_each(|chunk| {
        let mut rng = crate::util::blinding_rng::BlindingRng::new();
        for word in chunk {
            *word = F::sample(&mut rng).to_noncanonical_u64();
        }
    });
    words
}

/// Reads `count` elements back once submission `done` has executed.
fn read_fields<F: PrimeField64>(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    count: usize,
    done: &wgpu::SubmissionIndex,
    timing: &mut TimingTree,
) -> Vec<F> {
    ctx.read_after_with_timing(
        buffer,
        (count * 8) as u64,
        Some(done),
        Some(timing),
        |bytes| {
            bytes
                .as_chunks::<8>()
                .0
                .par_iter()
                .map(|chunk| F::from_noncanonical_u64(u64::from_le_bytes(*chunk)))
                .collect()
        },
    )
}

/// Reads `columns` columns of `len` elements back.
pub(crate) fn read_columns<F: PrimeField64>(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    columns: usize,
    len: usize,
) -> Vec<Vec<F>> {
    read_columns_with_timing(ctx, buffer, columns, len, None)
}

fn read_columns_with_timing<F: PrimeField64>(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    columns: usize,
    len: usize,
    timing: Option<&mut TimingTree>,
) -> Vec<Vec<F>> {
    ctx.read_with_timing(buffer, (columns * len * 8) as u64, timing, |bytes| {
        decode_columns(bytes, len)
    })
}

/// Columns of `len` little-endian field words.
fn decode_columns<F: PrimeField64>(bytes: &[u8], len: usize) -> Vec<Vec<F>> {
    bytes
        .par_chunks_exact(len * 8)
        .map(|column| {
            column
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| F::from_noncanonical_u64(u64::from_le_bytes(*chunk)))
                .collect()
        })
        .collect()
}

/// Coefficients already on the host, or their device-only shape.
enum HostCoefficients<F: Field> {
    Ready(Vec<PolynomialCoeffs<F>>),
    Device { columns: usize, degree: usize },
}

/// `PolynomialBatch::from_values` on the device.
#[cfg(test)]
pub(crate) fn from_values<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &std::sync::Arc<GpuContext>,
    values: Vec<PolynomialValues<F>>,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
) -> PolynomialBatch<F, C, D> {
    from_values_with_residency(
        ctx,
        values,
        rate_bits,
        blinding,
        cap_height,
        timing,
        ctx.options.leaf_storage(ctx.mappable),
        false,
    )
    .into_host()
}

pub(crate) fn from_values_with_residency<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &std::sync::Arc<GpuContext>,
    values: Vec<PolynomialValues<F>>,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    leaf_storage: LeafStorage,
    retain_values: bool,
) -> ProverPolynomialBatch<F, C, D> {
    let degree = values[0].len();
    let num_polys = values.len();
    let vals = timed!(timing, "gpu: upload values", {
        let columns = values
            .iter()
            .map(|v| v.values.as_slice())
            .collect::<Vec<_>>();
        upload_columns_with_timing(ctx, "values", &columns, Some(timing))
    });
    timed!(timing, "gpu: release input values", drop(values));
    from_device_values(
        ctx,
        vals,
        num_polys,
        degree,
        rate_bits,
        blinding,
        cap_height,
        timing,
        leaf_storage,
        retain_values,
    )
}

/// `PolynomialBatch::from_values` for `num_polys` columns of `degree`
/// values already on the device (column-major in `vals`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn from_device_values<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &std::sync::Arc<GpuContext>,
    vals: DeviceBuffer,
    num_polys: usize,
    degree: usize,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    leaf_storage: LeafStorage,
    retain_values: bool,
) -> ProverPolynomialBatch<F, C, D> {
    let log_d = log2_strict(degree) as u32;
    let (coeffs, work) = timed!(timing, "gpu: allocate IFFT buffers", {
        let coeffs = ctx.storage("coeffs", (num_polys * degree * 8) as u64);
        let work = if fft_plan(log_d).len() > 1 {
            ctx.storage("ifft work", (num_polys * degree * 8) as u64)
        } else {
            ctx.dummy_rw.clone()
        };
        (coeffs, work)
    });
    let encoder = timed!(timing, "gpu: encode IFFT", {
        let mut encoder = ctx.encoder("ifft");
        fft(
            ctx,
            &mut encoder,
            FftJob {
                log_n: log_d,
                columns: num_polys as u32,
                input: &vals,
                in_len: degree as u32,
                in_stride: degree as u32,
                data: &work,
                output: Some(&coeffs),
                shift_in: None,
                inverse: true,
                shift_out: None,
            },
        );
        encoder
    });
    let device_values = crate::gpu::DeviceCache::default();
    timed!(timing, "gpu: IFFT", ctx.submit_async(encoder));
    // Queue order keeps the scratch buffer intact for the IFFT even if the
    // pool hands it out again; releasing it here lowers peak memory.
    timed!(timing, "gpu: release IFFT work", drop(work));
    if retain_values {
        device_values.set(vals);
    } else {
        drop(vals);
    }
    commit(
        ctx,
        HostCoefficients::Device {
            columns: num_polys,
            degree,
        },
        coeffs,
        device_values,
        rate_bits,
        blinding,
        cap_height,
        timing,
        leaf_storage,
    )
}

/// Commits column-major device coefficients without a host copy. CPU opening
/// fallbacks materialize the coefficients only if needed.
pub(crate) fn from_device_coeffs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &std::sync::Arc<GpuContext>,
    coeffs: DeviceBuffer,
    columns: usize,
    degree: usize,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    leaf_storage: LeafStorage,
) -> ProverPolynomialBatch<F, C, D> {
    commit(
        ctx,
        HostCoefficients::Device { columns, degree },
        coeffs,
        crate::gpu::DeviceCache::default(),
        rate_bits,
        blinding,
        cap_height,
        timing,
        leaf_storage,
    )
}

/// `PolynomialBatch::from_coeffs` on the device.
#[cfg(test)]
pub(crate) fn from_coeffs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &std::sync::Arc<GpuContext>,
    polynomials: Vec<PolynomialCoeffs<F>>,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
) -> PolynomialBatch<F, C, D> {
    from_coeffs_with_residency(
        ctx,
        polynomials,
        rate_bits,
        blinding,
        cap_height,
        timing,
        ctx.options.leaf_storage(ctx.mappable),
    )
    .into_host()
}

pub(crate) fn from_coeffs_with_residency<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &std::sync::Arc<GpuContext>,
    polynomials: Vec<PolynomialCoeffs<F>>,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    leaf_storage: LeafStorage,
) -> ProverPolynomialBatch<F, C, D> {
    let coeffs = timed!(timing, "gpu: upload coefficients", {
        audit_coefficient_transfer(
            "upload",
            "commitment",
            polynomials.len(),
            polynomials[0].len(),
            || {
                let columns = polynomials
                    .iter()
                    .map(|p| p.coeffs.as_slice())
                    .collect::<Vec<_>>();
                upload_columns_with_timing(ctx, "coeffs", &columns, Some(timing))
            },
        )
    });
    commit(
        ctx,
        HostCoefficients::Ready(polynomials),
        coeffs,
        crate::gpu::DeviceCache::default(),
        rate_bits,
        blinding,
        cap_height,
        timing,
        leaf_storage,
    )
}

/// LDE, hashing and transpose of `coeffs` (the device copy of `polynomials`).
#[allow(clippy::too_many_arguments)]
fn commit<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    ctx: &std::sync::Arc<GpuContext>,
    host_coeffs: HostCoefficients<F>,
    coeffs: DeviceBuffer,
    device_values: crate::gpu::DeviceCache,
    rate_bits: usize,
    blinding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    leaf_storage: LeafStorage,
) -> ProverPolynomialBatch<F, C, D> {
    let (num_polys, degree) = match &host_coeffs {
        HostCoefficients::Ready(polynomials) => {
            let degree = polynomials[0].len();
            assert!(
                polynomials.iter().all(|p| p.len() == degree),
                "Polynomial degrees inconsistent"
            );
            (polynomials.len(), degree)
        }
        HostCoefficients::Device { columns, degree } => (*columns, *degree),
    };
    let log_n = (log2_strict(degree) + rate_bits) as u32;
    let n = 1usize << log_n;
    let salt_size = if blinding { SALT_SIZE } else { 0 };
    let columns = num_polys + salt_size;
    let lde = timed!(
        timing,
        "gpu: allocate LDE buffer",
        ctx.storage("lde", (columns * n * 8) as u64)
    );
    let encoder = timed!(timing, "gpu: encode LDE", {
        let mut encoder = ctx.encoder("lde");
        fft(
            ctx,
            &mut encoder,
            FftJob {
                log_n,
                columns: num_polys as u32,
                input: &coeffs,
                in_len: degree as u32,
                in_stride: degree as u32,
                data: &lde,
                output: None,
                shift_in: Some(GoldilocksField::coset_shift()),
                inverse: false,
                shift_out: None,
            },
        );
        encoder
    });
    // The FFT is submitted without waiting for it: the salt is sampled and
    // written while the device runs it (into columns it does not touch), and
    // the leaf hashing below queues behind it, so the device goes from one to
    // the other without waiting for the host.
    let salts = timed!(timing, "gpu: FFT and salt", {
        timed!(
            timing,
            "gpu: FFT",
            timed!(
                timing,
                "gpu: finish and submit commands",
                ctx.submit_async(encoder)
            )
        );
        salt_words::<F>(salt_size, n)
    });
    // The opening phase evaluates the polynomials from these coefficients.
    let device_coeffs = crate::gpu::DeviceCache::default();
    device_coeffs.set(coeffs.clone());
    timed!(timing, "gpu: pack and enqueue salt writes", {
        // The salt columns follow the polynomials' columns in the LDE buffer.
        ctx.write(&lde, (num_polys * n) as u64, &salts);
    });
    let rows = (leaf_storage != LeafStorage::Columns).then(|| {
        timed!(
            timing,
            "gpu: allocate leaf buffer",
            ctx.storage("leaves", (columns * n * 8) as u64)
        )
    });
    let tree = timed!(
        timing,
        "gpu: submit leaf hashing with row layout, and levels",
        build_tree(
            ctx,
            &lde,
            LeafSource::Columns {
                columns: columns as u32,
                log_n,
                arity_bits: 0,
            },
            log_n,
            cap_height as u32,
            rows.as_deref(),
        )
    );
    // The rows are complete once the leaves are hashed; they are read while
    // the levels above are still being hashed.
    let leaves = match leaf_storage {
        LeafStorage::Columns => MerkleLeaves::from_device_columns(
            ctx.clone(),
            lde.clone(),
            columns,
            log_n,
            0,
            tree.leaves_done.clone(),
        ),
        LeafStorage::Rows => MerkleLeaves::from_device(
            ctx.clone(),
            rows.expect("row buffer"),
            columns,
            n,
            tree.leaves_done.clone(),
        ),
        LeafStorage::Host => timed!(
            timing,
            "gpu: hash leaves and read rows",
            MerkleLeaves::new(
                read_fields::<F>(
                    ctx,
                    rows.as_ref().expect("row buffer"),
                    n * columns,
                    &tree.leaves_done,
                    timing
                ),
                columns,
                n
            )
        ),
    };
    // With retained leaves (within a proof) the digests stay on the device
    // too; the FRI queries read only their authentication paths.
    let (digests, cap) = if leaf_storage != LeafStorage::Host {
        let cap = timed!(
            timing,
            "gpu: hash levels and read cap",
            tree.read_cap::<F, C::Hasher>(ctx, Some(timing))
        );
        let digests = MerkleDigests::from_device::<F, C::Hasher>(
            ctx.clone(),
            tree.digests,
            tree.num_digests,
            tree.done,
        );
        (digests, cap)
    } else {
        let (digests, cap) = timed!(
            timing,
            "gpu: hash levels and read digests",
            tree.read_with_timing::<F, C::Hasher>(ctx, Some(timing))
        );
        (digests.into(), cap)
    };
    let device = crate::gpu::DeviceCache::default();
    device.set(lde);
    timed!(timing, "gpu: release blinding salts", drop(salts));
    let batch = PolynomialBatch {
        polynomials: Vec::new(),
        merkle_tree: MerkleTree {
            leaves,
            digests,
            cap: MerkleCap(cap),
        },
        degree_log: log2_strict(degree),
        rate_bits,
        blinding,
        device,
        device_coeffs,
        device_values,
    };
    match host_coeffs {
        HostCoefficients::Ready(polynomials) => ProverPolynomialBatch::from(PolynomialBatch {
            polynomials,
            ..batch
        }),
        HostCoefficients::Device { columns, degree } => {
            ProverPolynomialBatch::from_device(batch, ctx.clone(), coeffs, columns, degree)
        }
    }
}

/// `PolynomialValues::ifft` of each column on the device (for tests and benchmarks).
#[cfg(test)]
pub(crate) fn ifft_batch<F: PrimeField64>(ctx: &GpuContext, values: &[Vec<F>]) -> Vec<Vec<F>> {
    let degree = values[0].len();
    let log_d = log2_strict(degree) as u32;
    let columns = values.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let vals = upload_columns(ctx, "values", &columns);
    let coeffs = ctx.storage("coeffs", (values.len() * degree * 8) as u64);
    let work = ctx.storage("ifft work", (values.len() * degree * 8) as u64);
    let mut encoder = ctx.encoder("ifft");
    fft(
        ctx,
        &mut encoder,
        FftJob {
            log_n: log_d,
            columns: values.len() as u32,
            input: &vals,
            in_len: degree as u32,
            in_stride: degree as u32,
            data: &work,
            output: Some(&coeffs),
            shift_in: None,
            inverse: true,
            shift_out: None,
        },
    );
    ctx.submit(encoder);
    read_columns(ctx, &coeffs, values.len(), degree)
}

/// `coset_fft_with_options(shift, Some(rate_bits), ..)` of each column's
/// zero-padded coefficients on the device (for tests and benchmarks).
#[cfg(test)]
pub(crate) fn coset_lde_batch<F: PrimeField64>(
    ctx: &GpuContext,
    coeffs: &[Vec<F>],
    rate_bits: usize,
    shift: F,
) -> Vec<Vec<F>> {
    let degree = coeffs[0].len();
    let log_n = (log2_strict(degree) + rate_bits) as u32;
    let n = 1usize << log_n;
    let columns = coeffs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let input = upload_columns(ctx, "coeffs", &columns);
    let lde = ctx.storage("lde", (coeffs.len() * n * 8) as u64);
    let mut encoder = ctx.encoder("lde");
    fft(
        ctx,
        &mut encoder,
        FftJob {
            log_n,
            columns: coeffs.len() as u32,
            input: &input,
            in_len: degree as u32,
            in_stride: degree as u32,
            data: &lde,
            output: None,
            shift_in: Some(GoldilocksField::from_noncanonical_u64(
                shift.to_noncanonical_u64(),
            )),
            inverse: false,
            shift_out: None,
        },
    );
    ctx.submit(encoder);
    read_columns(ctx, &lde, coeffs.len(), n)
}

/// `coset_ifft(shift)` of each column on the device (for tests and benchmarks).
#[cfg(test)]
pub(crate) fn coset_ifft_batch<F: PrimeField64>(
    ctx: &GpuContext,
    values: &[Vec<F>],
    shift: F,
) -> Vec<Vec<F>> {
    let n = values[0].len();
    let log_n = log2_strict(n) as u32;
    let columns = values.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let vals = upload_columns(ctx, "values", &columns);
    let coeffs = ctx.storage("coeffs", (values.len() * n * 8) as u64);
    let work = ctx.storage("ifft work", (values.len() * n * 8) as u64);
    let shift_inv = GoldilocksField::from_noncanonical_u64(shift.to_noncanonical_u64()).inverse();
    let mut encoder = ctx.encoder("coset ifft");
    fft(
        ctx,
        &mut encoder,
        FftJob {
            log_n,
            columns: values.len() as u32,
            input: &vals,
            in_len: n as u32,
            in_stride: n as u32,
            data: &work,
            output: Some(&coeffs),
            shift_in: None,
            inverse: true,
            shift_out: Some(shift_inv),
        },
    );
    ctx.submit(encoder);
    read_columns(ctx, &coeffs, values.len(), n)
}

#[cfg(test)]
mod salt_tests {
    use super::*;
    use crate::field::types::Field64;

    /// Chunked salts are field elements of the requested shape, and no two
    /// chunks repeat (each has its own OS-seeded generator). Needs no GPU.
    #[test]
    fn salt_words_are_independent_field_elements() {
        let (columns, n) = (4, 1 << 17);
        let words = salt_words::<GoldilocksField>(columns, n);
        assert_eq!(words.len(), columns * n);
        assert!(words.iter().all(|&w| w < GoldilocksField::ORDER));
        let chunks = words.chunks(SALT_CHUNK).collect::<Vec<_>>();
        let mut heads = chunks.iter().map(|c| &c[..4]).collect::<Vec<_>>();
        heads.sort_unstable();
        heads.dedup();
        assert_eq!(
            heads.len(),
            chunks.len(),
            "two salt chunks share a generator stream"
        );
        // A second call draws fresh randomness.
        assert_ne!(salt_words::<GoldilocksField>(columns, n)[..8], words[..8]);
    }
}
