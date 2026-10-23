//! Differential tests of the GPU kernels against the CPU implementations.
//! Most adapter-dependent tests skip with a message when none is available;
//! hardware regression tests require a physical GPU.

mod arithmetic_grid;
mod inversion;
mod residency_matrix;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng as StdRng;

use super::shader::{self, Flavor};
use super::*;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::{Field, PrimeField64};
use crate::fri::oracle::{PolynomialBatch, SALT_SIZE};
use crate::hash::merkle_tree::{MerkleLeaves, MerkleTree};
use crate::hash::poseidon::{Poseidon, PoseidonHash};
use crate::util::timing::TimingTree;

type F = GoldilocksField;
type H = PoseidonHash;

/// Held by tests that disable the GPU for the whole process, and by tests
/// that must see it enabled.
pub(crate) static ENABLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ctx() -> Option<std::sync::Arc<GpuContext>> {
    let ctx = context();
    if ctx.is_none() {
        eprintln!("no GPU adapter; skipping");
    }
    ctx
}

#[cfg(feature = "gpu-profiling")]
#[test]
fn resident_merkle_query_paths() {
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(_available) = ctx() else { return };
    for force_staging in [false, true] {
        let mut ctx = GpuContext::new().expect("query test requires a GPU");
        if force_staging {
            ctx.mappable = false;
        }
        let words = (0..128 * 9).map(|i| i as u64).collect::<Vec<_>>();
        let source = ctx.upload("query test leaves", &words);
        let leaves = MerkleLeaves::new(
            words.iter().map(|&x| F::from_canonical_u64(x)).collect(),
            9,
            128,
        );
        for cap_height in [0, 1, 4, 7] {
            let expected = MerkleTree::<F, H>::new_cpu(leaves.clone(), cap_height);
            let tree = merkle::build_tree(
                &ctx,
                &source,
                merkle::LeafSource::Rows { leaf_len: 9 },
                7,
                cap_height as u32,
                None,
            );
            assert_eq!(tree.read_cap::<F, H>(&ctx, None), expected.cap.0);
            for indices in [vec![], vec![0, 127, 64, 63, 0], (0..128).collect()] {
                let actual = tree.read_query_paths::<F, H>(&ctx, 128, cap_height, &indices);
                let expected = indices
                    .iter()
                    .map(|&i| expected.prove(i).siblings)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "cap height {cap_height}");
            }
            // Full reads still agree after sparse reads and after repeated queries.
            let (digests, cap) = tree.read::<F, H>(&ctx);
            assert_eq!(digests.as_slice(), &*expected.digests);
            assert_eq!(cap, expected.cap.0);
            if let Some(t) = tree.gpu_timings(&ctx) {
                assert!(t.leaves_ms.is_finite() && t.leaves_ms >= 0.0);
                assert!(t.levels_ms.is_finite() && t.levels_ms >= 0.0);
            }
        }
    }
}

/// Field values at and around the boundaries, including noncanonical words.
fn boundary_words() -> Vec<u64> {
    vec![
        0,
        1,
        2,
        0xffff,
        0x10000,
        0xffff_fffe,
        0xffff_ffff,
        0x1_0000_0000,
        0x1_0000_0001,
        0xffff_ffff_ffff,
        0x8000_0000_0000_0000,
        GOLDILOCKS_ORDER - 2,
        GOLDILOCKS_ORDER - 1,
        GOLDILOCKS_ORDER,
        GOLDILOCKS_ORDER + 1,
        u64::MAX - 1,
        u64::MAX,
    ]
}

fn random_fields(rng: &mut StdRng, n: usize) -> Vec<F> {
    (0..n)
        .map(|_| F::from_canonical_u64(rng.next_u64() % GOLDILOCKS_ORDER))
        .collect()
}

#[test]
fn gpu_proof_of_work_finds_the_smallest_cpu_nonce() {
    use crate::hash::hashing::PlonkyPermutation;
    use crate::hash::poseidon::PoseidonPermutation;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let mut rng = StdRng::seed_from_u64(0x0f0f);
    let response_zeros = |state: &[F], position: usize, candidate: u64| {
        let mut p = PoseidonPermutation::<F>::new(state.iter().copied());
        p.set_elt(F::from_canonical_u64(candidate), position);
        p.permute();
        p.squeeze()[7].to_canonical_u64().leading_zeros()
    };
    let boundary = boundary_words()
        .into_iter()
        .take(12)
        .map(F::from_noncanonical_u64)
        .collect::<Vec<_>>();
    for (case, position) in [0usize, 3, 7, 8, 11].into_iter().enumerate() {
        let state = if case == 0 {
            boundary.clone()
        } else {
            random_fields(&mut rng, 12)
        };
        for min_zeros in [0u32, 6, 11, 16] {
            let found = crate::gpu::pow::search_on(&ctx, &state, position, min_zeros)
                .expect("device search finds a nonce");
            let smallest = (0u64..)
                .find(|&c| response_zeros(&state, position, c) >= min_zeros)
                .unwrap();
            assert_eq!(found, smallest, "position {position}, {min_zeros} zeros");
        }
    }
}

#[test]
fn shaders_validate() {
    for flavor in [Flavor::Native, Flavor::Portable] {
        let capabilities = match flavor {
            Flavor::Native => naga::valid::Capabilities::SHADER_INT64,
            Flavor::Portable => naga::valid::Capabilities::empty(),
        };
        for (name, source) in [
            ("hash", shader::hash_module(flavor)),
            ("pow", shader::pow_module(flavor)),
            ("fft", shader::fft_module(flavor)),
            ("fold", shader::fold_module(flavor)),
            ("open", shader::open_module(flavor)),
            ("combine", shader::combine_module(flavor)),
        ] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&source)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&source)));
        }
    }
}

#[test]
fn fft_plan_covers_all_stages() {
    for log_n in 0..=26 {
        let plan = shader::fft_plan(log_n);
        let mut next = 0;
        for (i, &(s0, stages, low)) in plan.iter().enumerate() {
            assert_eq!(s0, next);
            assert!(stages + low <= shader::FFT_TILE_BITS);
            assert!(low <= s0);
            assert!(i == 0 || stages > 0);
            next += stages;
        }
        assert_eq!(next, log_n);
    }
}

#[test]
fn permutation_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let mut states: Vec<[F; 12]> = boundary_words()
        .iter()
        .map(|&w| [F::from_noncanonical_u64(w); 12])
        .collect();
    for index in 0..12 {
        let mut state = [F::ZERO; 12];
        state[index] = F::from_noncanonical_u64(u64::MAX);
        states.push(state);
    }
    let mut rng = StdRng::seed_from_u64(0x504f534549444f4e);
    for _ in 0..512 {
        states.push(core::array::from_fn(|_| {
            F::from_noncanonical_u64(rng.next_u64())
        }));
    }
    let expected: Vec<[F; 12]> = states.iter().map(|&s| F::poseidon(s)).collect();
    let actual = ctx.permute_states(&states);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a, e, "state {i}: {:?}", states[i]);
        assert!(a.iter().all(|x| x.to_noncanonical_u64() < GOLDILOCKS_ORDER));
    }
}

#[test]
fn merkle_trees_match_cpu() {
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let mut rng = StdRng::seed_from_u64(0x4d45524b4c45);
    for log_n in [0u32, 1, 2, 4, 7, 10, 12, 14] {
        let n = 1usize << log_n;
        for leaf_len in [1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 135, 139] {
            if (log_n >= 12 && leaf_len > 17) || (log_n >= 14 && leaf_len > 5) {
                continue;
            }
            let mut leaves: Vec<Vec<F>> =
                (0..n).map(|_| random_fields(&mut rng, leaf_len)).collect();
            // Noncanonical and boundary words in the first leaves.
            for (leaf, &w) in leaves.iter_mut().zip(boundary_words().iter()) {
                leaf[0] = F::from_noncanonical_u64(w);
                if leaf_len > 1 {
                    leaf[leaf_len - 1] = F::from_noncanonical_u64(w ^ 0x5555);
                }
            }
            for cap_height in [0, 1, log_n.saturating_sub(1), log_n] {
                let cap_height = cap_height.min(log_n);
                let flat = MerkleLeaves::from_rows(leaves.clone());
                let expected = MerkleTree::<F, H>::new_cpu(flat.clone(), cap_height as usize);
                let (digests, cap) = merkle::build_from_rows::<F, H>(ctx, &flat, log_n, cap_height);
                assert_eq!(
                    digests.as_slice(),
                    &*expected.digests,
                    "log_n={log_n} leaf_len={leaf_len} cap_height={cap_height}"
                );
                assert_eq!(
                    cap, expected.cap.0,
                    "log_n={log_n} leaf_len={leaf_len} cap_height={cap_height}"
                );
            }
        }
    }
}

#[test]
fn merkle_tree_new_dispatches_to_gpu_above_threshold() {
    let Some(_ctx) = ctx() else { return };
    let log_n = crate::gpu::MIN_LOG_LEAVES.max(4);
    let mut rng = StdRng::seed_from_u64(7);
    let leaves: Vec<Vec<F>> = (0..1 << log_n)
        .map(|_| random_fields(&mut rng, 11))
        .collect();
    let gpu = MerkleTree::<F, H>::new(leaves.clone(), 3);
    let cpu = MerkleTree::<F, H>::new_cpu(MerkleLeaves::from_rows(leaves), 3);
    assert_eq!(gpu, cpu);
}

fn random_columns(rng: &mut StdRng, columns: usize, len: usize) -> Vec<Vec<F>> {
    (0..columns)
        .map(|c| {
            let mut column = random_fields(rng, len);
            // Boundary and noncanonical words in the first column.
            if c == 0 {
                for (x, &w) in column.iter_mut().zip(boundary_words().iter()) {
                    *x = F::from_noncanonical_u64(w);
                }
            }
            column
        })
        .collect()
}

#[test]
fn ifft_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let mut rng = StdRng::seed_from_u64(0x49464654);
    for log_n in [0u32, 1, 2, 3, 9, 10, 11, 13, 16, 17] {
        let columns = if log_n >= 16 { 3 } else { 5 };
        let values = random_columns(&mut rng, columns, 1 << log_n);
        let expected: Vec<Vec<F>> = values
            .iter()
            .map(|v| PolynomialValues::new(v.clone()).ifft().coeffs)
            .collect();
        let actual = lde::ifft_batch(ctx, &values);
        assert_eq!(actual, expected, "log_n={log_n}");
    }
}

#[test]
fn coset_lde_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let mut rng = StdRng::seed_from_u64(0x4c4445);
    for (log_d, rate_bits) in [
        (0u32, 1usize),
        (1, 3),
        (3, 2),
        (7, 3),
        (8, 3),
        (10, 1),
        (13, 3),
        (16, 3),
    ] {
        let columns = if log_d >= 13 { 3 } else { 5 };
        let coeffs = random_columns(&mut rng, columns, 1 << log_d);
        let shift = F::coset_shift();
        let expected: Vec<Vec<F>> = coeffs
            .iter()
            .map(|c| {
                PolynomialCoeffs::new(c.clone())
                    .lde(rate_bits)
                    .coset_fft_with_options(shift, Some(rate_bits), None)
                    .values
            })
            .collect();
        let actual = lde::coset_lde_batch(ctx, &coeffs, rate_bits, shift);
        assert_eq!(actual, expected, "log_d={log_d} rate_bits={rate_bits}");
    }
}

/// Zero-padded forward transforms through `fft_groups` directly, with and
/// without the coset shift and a separate output buffer, on shapes around
/// the tile and the skip cap: the skipped replication stages must leave
/// every transform identical to the CPU's. Each shape runs twice with
/// different inputs, so a stale pooled buffer cannot stand in for a slot
/// the kernel failed to write.
#[test]
fn zero_padded_fft_matches_cpu() {
    use crate::gpu::lde::{fft_groups, FftJob};
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let mut rng = StdRng::seed_from_u64(0x5a45524f);
    let shapes = [
        (0u32, 0usize),
        (0, 1),
        (0, 3),
        (0, 10),
        (0, 12),
        (1, 10),
        (2, 3),
        (6, 3),
        (6, 4),
        (7, 3),
        (7, 4),
        (8, 3),
        (3, 8),
        (9, 0),
        (10, 0),
        (10, 1),
        (11, 0),
        (13, 3),
        (16, 3),
    ];
    for (log_d, rate_bits) in shapes {
        for (shift, to_output) in [(false, false), (true, false), (true, true), (false, true)] {
            for _round in 0..2 {
                let columns = if log_d >= 13 { 2 } else { 3 };
                let log_n = log_d + rate_bits as u32;
                let n = 1usize << log_n;
                let degree = 1usize << log_d;
                let mut coeffs = random_columns(&mut rng, columns, degree);
                // Noncanonical words in the input too.
                for c in &mut coeffs {
                    c[0] = F::from_noncanonical_u64(u64::MAX);
                }
                let expected: Vec<Vec<F>> = coeffs
                    .iter()
                    .map(|c| {
                        let p = PolynomialCoeffs::new(c.clone()).lde(rate_bits);
                        if shift {
                            p.coset_fft_with_options(F::coset_shift(), Some(rate_bits), None)
                                .values
                        } else {
                            p.fft_with_options(Some(rate_bits), None).values
                        }
                    })
                    .collect();
                let refs = coeffs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                let input = lde::upload_columns(ctx, "zp coeffs", &refs);
                let data = ctx.storage("zp data", (columns * n * 8) as u64);
                let output = ctx.storage("zp output", (columns * n * 8) as u64);
                let mut encoder = ctx.encoder("zero padded fft");
                fft_groups(
                    ctx,
                    &mut encoder,
                    FftJob {
                        log_n,
                        columns: columns as u32,
                        input: &input,
                        in_len: degree as u32,
                        in_stride: degree as u32,
                        data: &data,
                        output: to_output.then_some(&output),
                        shift_in: shift.then(GoldilocksField::coset_shift),
                        inverse: false,
                        shift_out: None,
                    },
                    None,
                );
                ctx.submit(encoder);
                let actual: Vec<Vec<F>> =
                    lde::read_columns(ctx, if to_output { &output } else { &data }, columns, n);
                assert_eq!(
                    actual, expected,
                    "log_d={log_d} rate_bits={rate_bits} shift={shift} to_output={to_output}"
                );
            }
        }
    }
}

#[test]
fn coset_ifft_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let mut rng = StdRng::seed_from_u64(0x434f534554);
    for log_n in [1u32, 4, 10, 12, 15] {
        let values = random_columns(&mut rng, 3, 1 << log_n);
        let shift = F::coset_shift();
        let expected: Vec<Vec<F>> = values
            .iter()
            .map(|v| PolynomialValues::new(v.clone()).coset_ifft(shift).coeffs)
            .collect();
        let actual = lde::coset_ifft_batch(ctx, &values, shift);
        assert_eq!(actual, expected, "log_n={log_n}");
    }
}

#[test]
fn polynomial_batches_match_cpu() {
    use crate::plonk::config::PoseidonGoldilocksConfig;
    type C = PoseidonGoldilocksConfig;
    let Some(ctx) = ctx() else { return };
    let ctx = &ctx;
    let mut rng = StdRng::seed_from_u64(0x4241544348);
    for (log_d, rate_bits, columns, cap_height) in [
        (4usize, 3usize, 7usize, 2usize),
        (8, 3, 35, 4),
        (10, 2, 3, 0),
        (13, 3, 9, 4),
    ] {
        let values = random_columns(&mut rng, columns, 1 << log_d)
            .into_iter()
            .map(PolynomialValues::new)
            .collect::<Vec<_>>();
        let mut timing = TimingTree::default();
        let expected = PolynomialBatch::<F, C, 2>::from_values_cpu(
            values.clone(),
            rate_bits,
            false,
            cap_height,
            &mut timing,
            None,
        );
        let actual = lde::from_values::<F, C, 2>(
            ctx,
            values.clone(),
            rate_bits,
            false,
            cap_height,
            &mut timing,
        );
        assert_eq!(actual.polynomials, expected.polynomials, "log_d={log_d}");
        assert_eq!(
            actual.merkle_tree.leaves, expected.merkle_tree.leaves,
            "log_d={log_d}"
        );
        assert_eq!(
            actual.merkle_tree.digests, expected.merkle_tree.digests,
            "log_d={log_d}"
        );
        assert_eq!(
            actual.merkle_tree.cap, expected.merkle_tree.cap,
            "log_d={log_d}"
        );
        assert_eq!(actual, expected);

        let from_coeffs = lde::from_coeffs::<F, C, 2>(
            ctx,
            expected.polynomials.clone(),
            rate_bits,
            false,
            cap_height,
            &mut timing,
        );
        assert_eq!(from_coeffs, expected);

        // With blinding the salt is random: the unsalted leaves and the polynomials
        // must agree, and the tree must be the tree of the salted leaves.
        let blinded = lde::from_values::<F, C, 2>(
            ctx,
            values.clone(),
            rate_bits,
            true,
            cap_height,
            &mut timing,
        );
        assert_eq!(blinded.polynomials, expected.polynomials);
        assert!(blinded.blinding);
        for (b, e) in blinded
            .merkle_tree
            .leaves
            .rows()
            .zip(expected.merkle_tree.leaves.rows())
        {
            assert_eq!(b.len(), e.len() + SALT_SIZE);
            assert_eq!(&b[..e.len()], e);
        }
        let rebuilt = MerkleTree::<F, H>::new_cpu(blinded.merkle_tree.leaves.clone(), cap_height);
        assert_eq!(rebuilt, blinded.merkle_tree);
        // The salt columns hold this commitment's fresh salt: not zero, and
        // different in every row from a second commitment of the same values
        // (a misplaced or late salt write would leave stale pooled contents).
        let again = lde::from_values::<F, C, 2>(
            ctx,
            values.clone(),
            rate_bits,
            true,
            cap_height,
            &mut timing,
        );
        let width = expected.merkle_tree.leaves.leaf_len;
        for (r, (a, b)) in blinded
            .merkle_tree
            .leaves
            .rows()
            .zip(again.merkle_tree.leaves.rows())
            .enumerate()
        {
            assert!(
                a[width..].iter().any(|x| !x.is_zero()),
                "row {r}: zero salt"
            );
            assert_ne!(
                a[width..],
                b[width..],
                "row {r}: salt repeated across commitments"
            );
        }
    }
}

/// Upload buffers are reused without one upload overwriting another: two
/// threads each fill, copy and immediately refill pooled upload buffers of
/// the same size with distinct contents, and every destination reads back
/// exactly what was written into it.
#[test]
fn upload_staging_reuse_keeps_uploads_apart() {
    let Some(ctx) = ctx() else { return };
    let words = 1usize << 16;
    let bytes = (words * 8) as u64;
    let results = std::thread::scope(|scope| {
        let handles = (0..2u64)
            .map(|thread| {
                let ctx = &ctx;
                scope.spawn(move || {
                    (0..6u64)
                        .map(|round| {
                            let tag = (thread << 32) | (round << 16);
                            let staging = ctx.upload_staging(bytes);
                            {
                                let mut view = staging
                                    .buffer
                                    .slice(..bytes)
                                    .get_mapped_range_mut()
                                    .expect("mapped upload buffer");
                                let (cells, _) = view.slice(..).into_chunks::<8>();
                                for (i, cell) in cells.into_iter().enumerate() {
                                    cell.write((tag + i as u64).to_le_bytes());
                                }
                            }
                            let dst = ctx.storage("upload reuse test", bytes);
                            let done = ctx.finish_upload(staging, &dst, bytes);
                            (tag, dst, done)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (tag, dst, done) in results {
        let read = ctx.read_after(&dst, bytes, Some(&done), |b| {
            b.as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect::<Vec<_>>()
        });
        assert!(
            read.iter().enumerate().all(|(i, &w)| w == tag + i as u64),
            "upload {tag:#x} was overwritten"
        );
    }
}

#[test]
fn storage_pool_defaults_follow_memory_type() {
    assert_eq!(GpuOptions::default_pool_bytes(true), 4 << 30);
    assert_eq!(GpuOptions::default_pool_bytes(false), 0);
}

/// Out-of-range settings are refused before anything is stored, so this does
/// not change the configuration other tests run with.
#[test]
fn configure_rejects_invalid_settings() {
    let mut config = GpuConfig::default();
    config.proving_workers = 0;
    assert!(matches!(configure(config), Err(ConfigureError::Invalid(_))));
    for share in [-0.1, 1.5, f64::NAN] {
        let mut config = GpuConfig::default();
        config.quotient_cpu_share = share;
        assert!(matches!(configure(config), Err(ConfigureError::Invalid(_))));
    }
}

/// Once the global context has been created (even as "no GPU"), its settings
/// are fixed. Needs no GPU.
#[test]
fn configure_is_refused_once_the_context_exists() {
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = context();
    assert_eq!(
        configure(GpuConfig::default()),
        Err(ConfigureError::AlreadyInitialized)
    );
    assert_eq!(config(), GpuConfig::default());
}

#[test]
fn context_follows_its_config() {
    let Some(_) = ctx() else { return };
    let mut config = GpuConfig::default();
    config.mode = ExecutionMode::Resident;
    config.proving_workers = 3;
    config.pool_bytes = Some(3 << 20);
    config.quotient_cpu_share = 0.25;
    let ctx = GpuContext::with_config(&config).unwrap();
    assert_eq!(ctx.options.mode, ExecutionMode::Resident);
    assert_eq!(ctx.options.proving_workers, 3);
    assert_eq!(ctx.options.retained_bytes, 6 << 30);
    assert_eq!(ctx.options.pool_bytes, 3 << 20);
    assert_eq!(*ctx.quotient_cpu_share.lock().unwrap(), 0.25);
    drop(ctx);
    config.adapter_name = Some("no adapter has this name".into());
    assert!(GpuContext::with_config(&config).is_err());
    config.adapter_name = None;
    config.proving_workers = 0;
    assert!(GpuContext::with_config(&config).is_err());
}

#[test]
fn storage_pool_reuses_buffers() {
    let Some(_) = ctx() else { return };
    // A private context and explicit policy keep parallel tests from
    // changing this pool's contents or limit.
    let mut ctx = GpuContext::new().unwrap();
    ctx.options.pool_bytes = 3 << 20;
    ctx.storage_pool = Arc::new(StoragePool {
        free: Mutex::new(Vec::new()),
        limit: ctx.options.pool_bytes,
    });
    let before = ctx.storage_pool.bytes();
    let a = ctx.storage("pool test", 1 << 20);
    let raw: wgpu::Buffer = (*a).clone();
    let clone = a.clone();
    drop(a);
    // Still held by the clone: not in the pool.
    assert_eq!(ctx.storage_pool.bytes(), before);
    drop(clone);
    assert_eq!(ctx.storage_pool.bytes(), before + (1 << 20));
    // A different size creates a new buffer; the same size reuses the old one.
    let b = ctx.storage("pool test other", 1 << 21);
    assert!(*b != raw);
    let c = ctx.storage("pool test again", 1 << 20);
    assert!(*c == raw);
    assert_eq!(ctx.storage_pool.bytes(), before);
    // Buffers created by `upload` do not enter the storage pool.
    let d = ctx.upload("pool test upload", &[1, 2]);
    drop(d);
    assert_eq!(ctx.storage_pool.bytes(), before);
    drop((b, c));
    assert_eq!(ctx.storage_pool.bytes(), before + (3 << 20));
    // Oversized buffers are not retained; accepting another buffer evicts
    // old entries rather than exceeding the configured budget.
    drop(ctx.storage("oversized", 4 << 20));
    assert_eq!(ctx.storage_pool.bytes(), 3 << 20);
    drop(ctx.storage("eviction", 3 << 20));
    assert_eq!(ctx.storage_pool.bytes(), 3 << 20);
    ctx.options.pool_bytes = 0;
    ctx.storage_pool = Arc::new(StoragePool {
        free: Mutex::new(Vec::new()),
        limit: 0,
    });
    drop(ctx.storage("disabled", 1 << 20));
    assert_eq!(ctx.storage_pool.bytes(), 0);
}

#[test]
fn fri_commit_phase_matches_cpu() {
    use crate::field::extension::quadratic::QuadraticExtension;
    use crate::fri::prover::fri_committed_trees_cpu;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::fri::{FriConfig, FriParams};
    use crate::iop::challenger::Challenger;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    type C = PoseidonGoldilocksConfig;
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(_ctx) = ctx() else { return };
    let mut rng = StdRng::seed_from_u64(0x465249);
    for (degree_bits, arities, cap_height) in [
        (10usize, vec![4usize, 4], 2usize),
        (12, vec![4, 4, 2], 4),
        (13, vec![3, 3, 3, 1], 1),
        (16, vec![4, 4, 4], 4),
    ] {
        let rate_bits = 3;
        let n = 1 << (degree_bits + rate_bits);
        let mut coeffs = vec![QuadraticExtension::<F>::ZERO; n];
        for c in coeffs.iter_mut().take(1 << degree_bits) {
            *c = QuadraticExtension([
                F::from_noncanonical_u64(rng.next_u64()),
                F::from_noncanonical_u64(rng.next_u64()),
            ]);
        }
        let coeffs = PolynomialCoeffs::new(coeffs);
        let values = coeffs.coset_fft(F::coset_shift().into());
        let fri_params = FriParams {
            config: FriConfig {
                rate_bits,
                cap_height,
                proof_of_work_bits: 1,
                reduction_strategy: FriReductionStrategy::Fixed(arities.clone()),
                num_query_rounds: 2,
            },
            hiding: false,
            degree_bits,
            reduction_arity_bits: arities.clone(),
        };
        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_elements(&[F::ONE, F::TWO]);
        let mut cpu_challenger = challenger.clone();
        let (cpu_trees, cpu_final) = fri_committed_trees_cpu::<F, C, 2>(
            coeffs.clone(),
            values.clone(),
            &mut cpu_challenger,
            &fri_params,
            Some(1 << 6),
            Some(arities.len() + 1),
        );
        let (gpu_trees, gpu_final) = fri::try_committed_trees::<F, C, 2>(
            &coeffs,
            &values,
            &mut challenger,
            &fri_params,
            Some(1 << 6),
            Some(arities.len() + 1),
        )
        .expect("GPU takes the instance");
        assert_eq!(gpu_trees.len(), cpu_trees.len());
        for (layer, (g, c)) in gpu_trees.iter().zip(&cpu_trees).enumerate() {
            assert_eq!(g.cap, c.cap, "degree_bits={degree_bits} layer {layer} cap");
            assert_eq!(
                g.digests, c.digests,
                "degree_bits={degree_bits} layer {layer} digests"
            );
            assert_eq!(
                g.leaves, c.leaves,
                "degree_bits={degree_bits} layer {layer} leaves"
            );
        }
        assert_eq!(
            gpu_final, cpu_final,
            "degree_bits={degree_bits} final polynomial"
        );
        assert_eq!(
            challenger.get_n_challenges(4),
            cpu_challenger.get_n_challenges(4),
            "degree_bits={degree_bits} challenger state"
        );
        assert!(crate::gpu::MIN_LOG_LEAVES <= (degree_bits + rate_bits) as u32);
    }
}

/// A circuit of about `rows` Poseidon hashes.
pub(crate) fn hash_chain_circuit(
    rows: usize,
) -> (
    crate::plonk::circuit_data::CircuitData<F, crate::plonk::config::PoseidonGoldilocksConfig, 2>,
    crate::iop::witness::PartialWitness<F>,
) {
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    let config = CircuitConfig::standard_recursion_zk_config();
    let mut builder = CircuitBuilder::<F, 2>::new(config);
    let input = builder.add_virtual_target();
    let mut state = vec![input; 4];
    for _ in 0..rows {
        let hash = builder.hash_n_to_hash_no_pad::<H>(state.clone());
        state = hash.elements.to_vec();
    }
    builder.register_public_input(input);
    builder.register_public_inputs(&state);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u64(42)).unwrap();
    (builder.build::<PoseidonGoldilocksConfig>(), pw)
}

#[test]
fn proofs_verify_and_circuit_data_matches_cpu() {
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::proof::ProofWithPublicInputs;
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(_ctx) = ctx() else { return };
    let rows = 1usize << crate::gpu::MIN_LOG_LEAVES.saturating_sub(2).max(7);
    let (gpu_data, pw) = hash_chain_circuit(rows);
    assert!(
        gpu_data.common.degree_bits() + gpu_data.common.config.fri_config.rate_bits
            >= crate::gpu::MIN_LOG_LEAVES as usize
    );
    let gpu_proof = gpu_data.prove(pw.clone()).unwrap();
    set_enabled(false);
    let (cpu_data, pw) = hash_chain_circuit(rows);
    let cpu_proof = cpu_data.prove(pw).unwrap();
    set_enabled(true);
    assert_eq!(gpu_data.common, cpu_data.common);
    assert_eq!(gpu_data.verifier_only, cpu_data.verifier_only);
    assert_eq!(gpu_proof.public_inputs, cpu_proof.public_inputs);
    // Each proof verifies with the other build's verifier data.
    cpu_data.verify(gpu_proof.clone()).unwrap();
    gpu_data.verify(cpu_proof).unwrap();
    // A corrupted GPU proof fails like a corrupted CPU proof.
    let bytes = gpu_proof.to_bytes();
    let mut corrupted = bytes.clone();
    let index = corrupted.len() / 2;
    corrupted[index] ^= 1;
    let corrupted = ProofWithPublicInputs::<F, PoseidonGoldilocksConfig, 2>::from_bytes(
        corrupted,
        &gpu_data.common,
    )
    .unwrap();
    assert!(cpu_data.verify(corrupted).is_err());
    let mut wrong_inputs = gpu_proof;
    wrong_inputs.public_inputs[0] = F::ZERO;
    assert!(cpu_data.verify(wrong_inputs).is_err());
}

/// A circuit using the arithmetic, exponentiation, random access, base sum,
/// reducing, coset interpolation and Poseidon gates.
pub(crate) fn mixed_circuit() -> (
    crate::plonk::circuit_data::CircuitData<F, crate::plonk::config::PoseidonGoldilocksConfig, 2>,
    crate::iop::witness::PartialWitness<F>,
) {
    use crate::field::extension::quadratic::QuadraticExtension;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::util::reducing::ReducingFactorTarget;
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, 2>::new(config);
    let input = builder.add_virtual_target();
    let mut acc = input;
    for i in 0..40u64 {
        let c = builder.constant(F::from_canonical_u64(i + 3));
        let m = builder.mul(acc, c);
        acc = builder.add(m, input);
        acc = builder.exp_u64(acc, 5 + i % 7);
        let small = builder.mul_const(F::from_canonical_u64(i + 1), input);
        let bits = builder.split_le(small, 40);
        let packed = builder.le_sum(bits.iter().take(20));
        acc = builder.add(acc, packed);
        let v = (0..(1 << (1 + i % 4)))
            .map(|k| builder.mul_const(F::from_canonical_u64(k + 1), acc))
            .collect::<Vec<_>>();
        let index = builder.constant(F::from_canonical_u64(i % v.len() as u64));
        acc = builder.random_access(index, v);
        let ext = builder.convert_to_ext(acc);
        let sq = builder.mul_extension(ext, ext);
        let alpha = builder.constant_extension(QuadraticExtension([
            F::from_canonical_u64(7 + i),
            F::from_canonical_u64(11 + i),
        ]));
        let mut rf = ReducingFactorTarget::new(alpha);
        let terms = (0..(12 + (i % 30) as usize))
            .map(|_| sq)
            .collect::<Vec<_>>();
        let reduced = rf.reduce(&terms, &mut builder);
        let base_terms = (0..(20 + (i % 30) as usize))
            .map(|_| acc)
            .collect::<Vec<_>>();
        let reduced_base = rf.reduce_base(&base_terms, &mut builder);
        let s = builder.add_extension(reduced, reduced_base);
        acc = builder.add(s.0[0], s.0[1]);
        let hash = builder.hash_n_to_hash_no_pad::<H>(vec![acc, input, acc]);
        acc = hash.elements[0];
    }
    let shift = builder.constant(F::from_canonical_u64(5));
    let values = (0..16)
        .map(|k| {
            builder.constant_extension(QuadraticExtension([F::from_canonical_u64(k), F::ZERO]))
        })
        .collect::<Vec<_>>();
    let point = builder.convert_to_ext(acc);
    let interpolated = builder.interpolate_coset(
        crate::gates::coset_interpolation::CosetInterpolationGate::<F, 2>::with_max_degree(4, 6),
        shift,
        &values,
        point,
    );
    acc = builder.add(interpolated.0[0], interpolated.0[1]);
    builder.register_public_input(input);
    builder.register_public_input(acc);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u64(9)).unwrap();
    (builder.build::<PoseidonGoldilocksConfig>(), pw)
}

#[test]
fn mixed_circuit_proves_on_gpu() {
    use crate::plonk::config::PoseidonGoldilocksConfig;
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(_) = ctx() else { return };
    let (gpu_data, pw) = mixed_circuit();
    let gpu_proof = gpu_data.prove(pw.clone()).unwrap();
    set_enabled(false);
    let (cpu_data, pw) = mixed_circuit();
    let cpu_proof = cpu_data.prove(pw).unwrap();
    set_enabled(true);
    assert_eq!(gpu_data.common, cpu_data.common);
    assert_eq!(gpu_data.verifier_only, cpu_data.verifier_only);
    cpu_data.verify(gpu_proof).unwrap();
    gpu_data.verify(cpu_proof).unwrap();
    let ids = gpu_data
        .common
        .gates
        .iter()
        .map(|g| g.0.id())
        .collect::<Vec<_>>();
    eprintln!("mixed circuit gates: {ids:?}");
    let _ = PoseidonGoldilocksConfig;
}

/// A gate exercising quintic product fusion with two operands and with a
/// square. Each coefficient is a fold of plain products plus a fold of
/// products scaled by the modulus constant. Wires: x, y, x * y, x * x.
#[derive(Debug, Clone)]
struct QuinticMulTestGate {
    offset: usize,
}

impl QuinticMulTestGate {
    const W: u64 = crate::gpu::trace::QMUL_W;

    fn product<T: crate::field::types::Field>(x: &[T; 5], y: &[T; 5]) -> [T; 5] {
        let w = T::from_canonical_u64(Self::W);
        core::array::from_fn(|k| {
            let mut acc = T::ZERO;
            for i in 0..5 {
                let j = (k + 5 - i) % 5;
                let term = x[i] * y[j];
                acc += if i + j == k { term } else { term * w };
            }
            acc
        })
    }

    fn product_circuit(
        builder: &mut crate::plonk::circuit_builder::CircuitBuilder<F, 2>,
        x: &[crate::iop::ext_target::ExtensionTarget<2>; 5],
        y: &[crate::iop::ext_target::ExtensionTarget<2>; 5],
    ) -> [crate::iop::ext_target::ExtensionTarget<2>; 5] {
        let zero = builder.zero_extension();
        let w = F::from_canonical_u64(Self::W);
        core::array::from_fn(|k| {
            let term1 = (0..=k)
                .map(|i| builder.mul_extension(x[i], y[k - i]))
                .collect::<Vec<_>>()
                .into_iter()
                .reduce(|sum, summand| builder.add_extension(sum, summand))
                .unwrap();
            if k == 4 {
                return term1;
            }
            let term2 = (k + 1..5)
                .map(|i| builder.arithmetic_extension(w, F::ZERO, x[i], y[5 + k - i], zero))
                .collect::<Vec<_>>()
                .into_iter()
                .reduce(|sum, summand| builder.add_extension(sum, summand))
                .unwrap();
            builder.add_extension(term1, term2)
        })
    }
}

impl crate::gates::gate::Gate<F, 2> for QuinticMulTestGate {
    fn id(&self) -> String {
        format!("QuinticMulTestGate({})", self.offset)
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &crate::plonk::circuit_data::CommonCircuitData<F, 2>,
    ) -> crate::util::serialization::IoResult<()> {
        use crate::util::serialization::Write;
        dst.write_usize(self.offset)
    }

    fn deserialize(
        src: &mut crate::util::serialization::Buffer,
        _common_data: &crate::plonk::circuit_data::CommonCircuitData<F, 2>,
    ) -> crate::util::serialization::IoResult<Self> {
        use crate::util::serialization::Read;
        Ok(Self {
            offset: src.read_usize()?,
        })
    }

    fn eval_unfiltered(
        &self,
        vars: crate::plonk::vars::EvaluationVars<F, 2>,
    ) -> Vec<<F as crate::field::extension::Extendable<2>>::Extension> {
        let x = core::array::from_fn(|i| vars.local_wires[self.offset + i]);
        let y = core::array::from_fn(|i| vars.local_wires[self.offset + 5 + i]);
        let xy = Self::product(&x, &y);
        let xx = Self::product(&x, &x);
        (0..5)
            .map(|k| xy[k] - vars.local_wires[self.offset + 10 + k])
            .chain((0..5).map(|k| xx[k] - vars.local_wires[self.offset + 15 + k]))
            .collect()
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut crate::plonk::circuit_builder::CircuitBuilder<F, 2>,
        vars: crate::plonk::vars::EvaluationTargets<2>,
    ) -> Vec<crate::iop::ext_target::ExtensionTarget<2>> {
        let x = core::array::from_fn(|i| vars.local_wires[self.offset + i]);
        let y = core::array::from_fn(|i| vars.local_wires[self.offset + 5 + i]);
        let xy = Self::product_circuit(builder, &x, &y);
        let xx = Self::product_circuit(builder, &x, &x);
        let mut constraints = Vec::new();
        for k in 0..5 {
            constraints.push(builder.sub_extension(xy[k], vars.local_wires[self.offset + 10 + k]));
        }
        for k in 0..5 {
            constraints.push(builder.sub_extension(xx[k], vars.local_wires[self.offset + 15 + k]));
        }
        constraints
    }

    fn generators(
        &self,
        _row: usize,
        _local_constants: &[F],
    ) -> Vec<crate::iop::generator::WitnessGeneratorRef<F, 2>> {
        Vec::new()
    }

    fn num_wires(&self) -> usize {
        self.offset + 20
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        10
    }
}

/// The test gate's two quintic products (one a square, whose symmetric
/// products the tracer hash-conses) each become one `Qmul` group and one op.
#[test]
fn quintic_products_fuse() {
    use crate::plonk::circuit_data::CircuitConfig;
    let config = CircuitConfig::standard_recursion_config();
    let gate = crate::gates::gate::GateRef::new(QuinticMulTestGate { offset: 0 });
    let mut trace = trace::trace_gate(&gate, &config, 0).unwrap();
    trace::fuse_quintic_products(&mut trace);
    assert_eq!(trace.dag.qmul_groups.len(), 2);
    let (ops, _) = quotient::gate_program(&trace, 0, &[], 0).unwrap();
    let qmuls = ops
        .iter()
        .filter(|op| op[0] & quotient::BASE_MASK == 17)
        .count();
    assert_eq!(qmuls, 2);
    // Nothing of the expanded products survives: wire reads, the two
    // products, ten weighted constraints and their subtractions, the filter.
    assert!(ops.len() < 40, "{} ops", ops.len());
}

/// Packed quintic operands have an eight-bit index; wider valid gates must
/// decline fused compilation so the compiler can retry ordinary GPU arithmetic.
#[test]
fn quintic_program_rejects_wide_indices() {
    use crate::plonk::circuit_data::CircuitConfig;
    let config = CircuitConfig::standard_recursion_config();
    let gate = crate::gates::gate::GateRef::new(QuinticMulTestGate { offset: 0 });
    let mut trace = trace::trace_gate(&gate, &config, 0).unwrap();
    trace::fuse_quintic_products(&mut trace);
    // This wire participates in a recognized quintic product. Index 255
    // still fits; index 256 must not panic or be truncated to zero.
    let wire = trace
        .dag
        .nodes
        .iter()
        .position(|n| matches!(n, trace::Node::Wire(0)))
        .unwrap();
    trace.dag.nodes[wire] = trace::Node::Wire(255);
    assert!(quotient::gate_program(&trace, 0, &[], 0).is_some());
    trace.dag.nodes[wire] = trace::Node::Wire(256);
    assert!(quotient::gate_program(&trace, 0, &[], 0).is_none());

    // Keep coefficients live across many products to exceed the destination
    // register limit even though every wire index fits in eight bits.
    let mut dag = trace::Dag::<F>::new();
    let mut first = Vec::new();
    let mut rest = Vec::new();
    for i in 0..60 {
        let a = dag.push(trace::Node::Wire(2 * i));
        let b = dag.push(trace::Node::Wire(2 * i + 1));
        let ab = dag.mul(a, b);
        let scaled = dag.scale(ab, F::from_canonical_u64(trace::QMUL_W));
        // Expand [a; 5] * [b; 5] in the same fold order as the real gates.
        let product: [trace::NodeId; 5] = core::array::from_fn(|k| {
            let mut plain = ab;
            for _ in 0..k {
                plain = dag.add(plain, ab);
            }
            if k == 4 {
                return plain;
            }
            let mut wrapped = scaled;
            for _ in k + 2..5 {
                wrapped = dag.add(wrapped, scaled);
            }
            dag.add(plain, wrapped)
        });
        first.push(product[0]);
        rest.extend_from_slice(&product[1..]);
    }
    first.extend(rest);
    let trace = trace::GateTrace {
        dag,
        constraints: first,
    };
    let fused = trace::quintic_fused_trace(&trace).unwrap();
    assert_eq!(fused.dag.qmul_groups.len(), 60);
    assert!(quotient::gate_program(&fused, 0, &[], 0).is_none());
    let gates = [quotient::TracedGate {
        name: "register pressure".into(),
        trace,
        selector_index: 0,
        filter_factors: vec![],
    }];
    let program =
        quotient::compile_program(&gates, 0, 2).expect("unfused register allocation fits");
    assert!(!program.uses_qmul);
    let wires: Vec<F> = (1..=120).map(F::from_canonical_usize).collect();
    let alphas = [F::from_canonical_u64(7), F::from_canonical_u64(11)];
    let constraints = trace::evaluate(&gates[0].trace, &wires, &[F::ZERO], &[F::ZERO; 4]);
    let expected: Vec<F> = alphas
        .iter()
        .map(|alpha| {
            constraints
                .iter()
                .enumerate()
                .map(|(k, &c)| alpha.exp_u64(k as u64) * c)
                .sum()
        })
        .collect();
    assert_eq!(
        quotient::run_program_host(
            &program.ops,
            program.registers as usize,
            &wires,
            &[F::ZERO],
            &[F::ZERO; 4],
            &alphas
        ),
        expected
    );
}

#[test]
fn quintic_retry_discards_partial_program() {
    use crate::gates::gate::GateRef;
    use crate::plonk::circuit_data::CircuitConfig;
    let mut config = CircuitConfig::standard_recursion_config();
    config.num_wires = 276;
    // The first gate compiles successfully with QMUL. The second forces a
    // retry: neither the first gate's QMULs nor partial second-gate ops may
    // leak into the replacement program.
    let gates: Vec<_> = [0, 256]
        .into_iter()
        .map(|offset| {
            let gate = GateRef::new(QuinticMulTestGate { offset });
            let trace = trace::trace_gate(&gate, &config, 0).unwrap();
            assert!(trace.dag.qmul_groups.is_empty());
            assert_eq!(
                trace::quintic_fused_trace(&trace)
                    .unwrap()
                    .dag
                    .qmul_groups
                    .len(),
                2
            );
            quotient::TracedGate {
                name: gate.0.id(),
                trace,
                selector_index: 0,
                filter_factors: vec![3, 5],
            }
        })
        .collect();
    let program = quotient::compile_program(&gates, 7, 2).expect("unfused GPU program");
    assert!(!program.uses_qmul);
    assert!(!quotient::uses_qmul(&program.ops));
    assert!(gates.iter().all(|g| g.trace.dag.qmul_groups.is_empty()));
    let mut rng = StdRng::seed_from_u64(0x5245545259);
    for _ in 0..4 {
        let wires = random_fields(&mut rng, config.num_wires);
        let constants = random_fields(&mut rng, 1);
        let pih = random_fields(&mut rng, 4);
        let alphas = random_fields(&mut rng, 2);
        let filter =
            (F::from_canonical_u64(3) - constants[0]) * (F::from_canonical_u64(5) - constants[0]);
        let expected: Vec<F> = alphas
            .iter()
            .map(|&alpha| {
                gates
                    .iter()
                    .map(|gate| {
                        trace::evaluate(&gate.trace, &wires, &constants, &pih)
                            .iter()
                            .enumerate()
                            .map(|(k, &c)| filter * alpha.exp_u64(7 + k as u64) * c)
                            .sum::<F>()
                    })
                    .sum()
            })
            .collect();
        assert_eq!(
            quotient::run_program_host(
                &program.ops,
                program.registers as usize,
                &wires,
                &constants,
                &pih,
                &alphas
            ),
            expected
        );
    }
    // Unrelated hardware limits still decline GPU compilation.
    assert!(quotient::compile_program(&gates, 7, 5).is_none());
}

#[test]
fn quintic_variants_prove_on_gpu() {
    use crate::gates::noop::NoopGate;
    use crate::iop::target::Target;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    for offset in [0, 256] {
        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = config.num_wires.max(offset + 20);
        let mut builder = CircuitBuilder::<F, 2>::new(config);
        let row = builder.add_gate(QuinticMulTestGate { offset }, vec![]);
        while builder.num_gates() < (1 << crate::gpu::MIN_LOG_LEAVES) {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<PoseidonGoldilocksConfig>();
        let program = ctx
            .quotient_program(&data.common)
            .expect("GPU quotient program");
        assert_eq!(program.uses_qmul, offset == 0);
        for seed in [1, 2] {
            let mut rng = StdRng::seed_from_u64(seed);
            let x: [F; 5] = random_fields(&mut rng, 5).try_into().unwrap();
            let y: [F; 5] = random_fields(&mut rng, 5).try_into().unwrap();
            let xy = QuinticMulTestGate::product(&x, &y);
            let xx = QuinticMulTestGate::product(&x, &x);
            let mut pw = PartialWitness::new();
            for (i, value) in x.into_iter().chain(y).chain(xy).chain(xx).enumerate() {
                pw.set_target(Target::wire(row, offset + i), value).unwrap();
            }
            data.verify(data.prove(pw).unwrap()).unwrap();
        }
    }
}

#[test]
fn quotient_shader_variants_and_cache() {
    for flavor in [Flavor::Native, Flavor::Portable] {
        for uses_qmul in [false, true] {
            let source = quotient::kernel_source(flavor, 64, 2, uses_qmul);
            assert_eq!(source.contains("fn qmul("), uses_qmul);
            assert_eq!(source.contains("case 17u:"), uses_qmul);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let capabilities = match flavor {
                Flavor::Native => naga::valid::Capabilities::SHADER_INT64,
                Flavor::Portable => naga::valid::Capabilities::empty(),
            };
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap();
        }
    }
    let Some(ctx) = ctx() else { return };
    let plain = ctx.quotient_kernel(64, 2, false);
    let quintic = ctx.quotient_kernel(64, 2, true);
    assert!(!Arc::ptr_eq(&plain, &quintic));
    assert!(Arc::ptr_eq(&plain, &ctx.quotient_kernel(64, 2, false)));
    assert!(Arc::ptr_eq(&quintic, &ctx.quotient_kernel(64, 2, true)));
}

pub(crate) fn traced_gates() -> Vec<crate::gates::gate::GateRef<F, 2>> {
    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
    use crate::gates::base_sum::BaseSumGate;
    use crate::gates::constant::ConstantGate;
    use crate::gates::coset_interpolation::CosetInterpolationGate;
    use crate::gates::exponentiation::ExponentiationGate;
    use crate::gates::gate::GateRef;
    use crate::gates::multiplication_extension::MulExtensionGate;
    use crate::gates::poseidon::PoseidonGate;
    use crate::gates::poseidon_mds::PoseidonMdsGate;
    use crate::gates::public_input::PublicInputGate;
    use crate::gates::random_access::RandomAccessGate;
    use crate::gates::reducing::ReducingGate;
    use crate::gates::reducing_extension::ReducingExtensionGate;
    use crate::plonk::circuit_data::CircuitConfig;
    let config = CircuitConfig::standard_recursion_config();
    let mut gates = vec![
        GateRef::new(ArithmeticGate::new_from_config(&config)),
        GateRef::new(ArithmeticExtensionGate::<2>::new_from_config(&config)),
        GateRef::new(MulExtensionGate::<2>::new_from_config(&config)),
        GateRef::new(BaseSumGate::<2>::new(63)),
        GateRef::new(BaseSumGate::<4>::new(10)),
        GateRef::new(ConstantGate::new(2)),
        GateRef::new(PublicInputGate),
        GateRef::new(ReducingGate::<2>::new(43)),
        GateRef::new(ReducingExtensionGate::<2>::new(32)),
        GateRef::new(ExponentiationGate::<F, 2>::new_from_config(&config)),
        GateRef::new(CosetInterpolationGate::<F, 2>::with_max_degree(4, 6)),
        GateRef::new(CosetInterpolationGate::<F, 2>::with_max_degree(3, 3)),
        GateRef::new(PoseidonGate::<F, 2>::new()),
        GateRef::new(PoseidonMdsGate::<F, 2>::new()),
        GateRef::new(QuinticMulTestGate { offset: 0 }),
    ];
    for bits in 1..=6 {
        gates.push(GateRef::new(RandomAccessGate::<F, 2>::new_from_config(
            &config, bits,
        )));
    }
    gates
}

#[test]
fn gate_traces_match_cpu_evaluation() {
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::vars::EvaluationVarsBaseBatch;
    let config = CircuitConfig::standard_recursion_config();
    let mut rng = StdRng::seed_from_u64(0x5452414345);
    let batch = 8;
    for gate in traced_gates() {
        let mut trace = trace::trace_gate(&gate, &config, 0)
            .unwrap_or_else(|| panic!("{} is not traceable", gate.0.id()));
        let matched = trace::mds_matched_trace(&trace);
        let all = all_passes(&trace);
        trace::fuse_quintic_products(&mut trace);
        let folded = trace::negation_folded_trace(&trace);
        let num_wires = config.num_wires;
        let num_constants = gate.0.num_constants();
        let wires = random_columns(&mut rng, num_wires, batch);
        let constants = random_columns(&mut rng, num_constants.max(1), batch);
        let pih = crate::hash::hash_types::HashOut {
            elements: core::array::from_fn(|_| F::from_noncanonical_u64(rng.next_u64())),
        };
        // Column-major batches, as the prover lays them out.
        let wires_batch = wires.concat();
        let constants_batch = constants[..num_constants].concat();
        let expected = gate
            .0
            .eval_unfiltered_base_batch(EvaluationVarsBaseBatch::new(
                batch,
                &constants_batch,
                &wires_batch,
                &pih,
            ));
        let num_constraints = gate.0.num_constraints();
        for row in 0..batch {
            let row_wires = (0..num_wires).map(|i| wires[i][row]).collect::<Vec<_>>();
            let row_constants = (0..num_constants)
                .map(|j| constants[j][row])
                .collect::<Vec<_>>();
            let expected_row = (0..num_constraints)
                .map(|k| expected[k * batch + row])
                .collect::<Vec<_>>();
            let mut forms = vec![("fused", &trace), ("folded", &folded), ("all", &all)];
            if let Some(matched) = &matched {
                forms.push(("mds", matched));
            }
            for (form, trace) in forms {
                let actual = trace::evaluate(trace, &row_wires, &row_constants, &pih.elements);
                assert_eq!(actual, expected_row, "{} {form} row {row}", gate.0.id());
            }
        }
        eprintln!(
            "{}: {} nodes, {} constraints",
            gate.0.id(),
            trace.dag.nodes.len(),
            trace.constraints.len()
        );
    }
}

/// The linearized program of every gate, run on the host, gives the
/// alpha-weighted sum of the constraints the trace evaluates to.
#[test]
fn gate_programs_match_traces() {
    use crate::plonk::circuit_data::CircuitConfig;
    let config = CircuitConfig::standard_recursion_config();
    let mut rng = StdRng::seed_from_u64(0x50524f47);
    for (gate, fold) in traced_gates()
        .into_iter()
        .flat_map(|gate| [(gate.clone(), false), (gate, true)])
    {
        let mut trace = trace::trace_gate(&gate, &config, 0).unwrap();
        if fold {
            trace = all_passes(&trace);
        } else {
            trace::fuse_quintic_products(&mut trace);
        }
        let num_constants = gate.0.num_constants();
        // The gate's constants (traced at prefix 0), then the selector.
        let factors = [3u64, 5, crate::gates::selectors::UNUSED_SELECTOR as u64];
        let weight_offset = 7;
        let (ops, registers) =
            quotient::gate_program(&trace, num_constants as u32, &factors, weight_offset).unwrap();
        for _ in 0..4 {
            let wires = random_fields(&mut rng, config.num_wires);
            let row_constants = random_fields(&mut rng, num_constants + 1);
            let pih: Vec<F> = random_fields(&mut rng, 4);
            let alphas = random_fields(&mut rng, 2);
            let constraints = trace::evaluate(&trace, &wires, &row_constants, &pih);
            let selector = row_constants[num_constants];
            let filter = factors
                .iter()
                .map(|&f| F::from_canonical_u64(f) - selector)
                .product::<F>();
            let expected: Vec<F> = alphas
                .iter()
                .map(|&alpha| {
                    constraints
                        .iter()
                        .enumerate()
                        .map(|(k, &c)| filter * alpha.exp_u64(weight_offset as u64 + k as u64) * c)
                        .sum()
                })
                .collect();
            let actual = quotient::run_program_host(
                &ops,
                registers as usize,
                &wires,
                &row_constants,
                &pih,
                &alphas,
            );
            assert_eq!(actual, expected, "{} fold {fold}", gate.0.id());
        }
    }
}

/// The compiler's trace passes in its order: MDS matching, quintic fusion,
/// negation folding.
fn all_passes(trace: &trace::GateTrace<F>) -> trace::GateTrace<F> {
    let matched = trace::mds_matched_trace(trace);
    let base = matched.as_ref().unwrap_or(trace);
    let fused = trace::quintic_fused_trace(base);
    trace::negation_folded_trace(fused.as_ref().unwrap_or(base))
}

/// The MDS matcher replaces the PoseidonMds gate's written-out layers (one
/// per extension component) and matches nothing in the other gates.
#[test]
fn mds_layers_are_matched() {
    use crate::plonk::circuit_data::CircuitConfig;
    let config = CircuitConfig::standard_recursion_config();
    let factors = [3u64, 5, crate::gates::selectors::UNUSED_SELECTOR as u64];
    for gate in traced_gates() {
        let trace = trace::trace_gate(&gate, &config, 0).unwrap();
        let matched = trace::mds_matched_trace(&trace);
        let id = gate.0.id();
        if !id.starts_with("PoseidonMdsGate") {
            assert!(matched.is_none(), "{id}");
            continue;
        }
        let matched = matched.expect("PoseidonMds layers matched");
        assert_eq!(
            matched.dag.mds_groups.len(),
            trace.dag.mds_groups.len() + 2,
            "{id}"
        );
        let n = gate.0.num_constants() as u32;
        let (ops, _) = quotient::gate_program(&trace, n, &factors, 7).unwrap();
        let (matched_ops, _) = quotient::gate_program(&matched, n, &factors, 7).unwrap();
        eprintln!("{id}: {} -> {} ops", ops.len(), matched_ops.len());
        assert!(matched_ops.len() < ops.len() / 2);
    }
}

/// Folding removes every product by a constant just below the modulus and
/// never lengthens a gate's program.
#[test]
fn negation_folding_shrinks_programs() {
    use crate::plonk::circuit_data::CircuitConfig;
    let config = CircuitConfig::standard_recursion_config();
    let factors = [3u64, 5, crate::gates::selectors::UNUSED_SELECTOR as u64];
    let (mut before, mut after) = (0, 0);
    for gate in traced_gates() {
        let mut trace = trace::trace_gate(&gate, &config, 0).unwrap();
        trace::fuse_quintic_products(&mut trace);
        let folded = trace::negation_folded_trace(&trace);
        let near_modulus =
            |c: u64| c >= 1 << 32 && (-F::from_canonical_u64(c)).to_canonical_u64() < 1 << 32;
        let negative_products = folded
            .dag
            .nodes
            .iter()
            .filter(|node| {
                match **node {
                trace::Node::Mul(a, b) => [a, b].iter().any(|&x| {
                    matches!(folded.dag.nodes[x as usize], trace::Node::Const(c) if near_modulus(c))
                }),
                _ => false,
            }
            })
            .count();
        assert_eq!(negative_products, 0, "{}", gate.0.id());
        assert_eq!(folded.dag.qmul_groups.len(), trace.dag.qmul_groups.len());
        assert_eq!(folded.dag.mds_groups.len(), trace.dag.mds_groups.len());
        let n = gate.0.num_constants() as u32;
        let (ops, _) = quotient::gate_program(&trace, n, &factors, 7).unwrap();
        let (folded_ops, _) = quotient::gate_program(&folded, n, &factors, 7).unwrap();
        assert!(
            folded_ops.len() <= ops.len(),
            "{}: {} > {}",
            gate.0.id(),
            folded_ops.len(),
            ops.len()
        );
        eprintln!("{}: {} -> {} ops", gate.0.id(), ops.len(), folded_ops.len());
        before += ops.len();
        after += folded_ops.len();
    }
    eprintln!("all gates: {before} -> {after} ops");
    assert!(after < before);
}

#[test]
fn gate_kernels_match_cpu_evaluation() {
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::vars::EvaluationVarsBaseBatch;
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let config = CircuitConfig::standard_recursion_config();
    let mut rng = StdRng::seed_from_u64(0x4b45524e454c);
    let batch = 256;
    for gate in traced_gates() {
        let trace = all_passes(&trace::trace_gate(&gate, &config, 0).unwrap());
        let num_wires = config.num_wires;
        let num_constants = gate.0.num_constants();
        let num_constraints = gate.0.num_constraints();
        let stride = num_wires + num_constants;
        let wires = random_columns(&mut rng, num_wires, batch);
        let constants = random_columns(&mut rng, num_constants.max(1), batch);
        let pih = crate::hash::hash_types::HashOut {
            elements: core::array::from_fn(|_| F::from_noncanonical_u64(rng.next_u64())),
        };
        let expected = gate
            .0
            .eval_unfiltered_base_batch(EvaluationVarsBaseBatch::new(
                batch,
                &constants[..num_constants].concat(),
                &wires.concat(),
                &pih,
            ));
        let mut source = shader::prelude_with_mds(ctx.flavor);
        if !trace.dag.qmul_groups.is_empty() {
            source.push_str(shader::QMUL);
        }
        source.push_str(&format!(
            r#"
struct TParams {{ count: u32, stride: u32, num_wires: u32, pad: u32 }}
@group(0) @binding(0) var<uniform> tp: TParams;
@group(0) @binding(1) var<storage, read> src: array<Field>;
@group(0) @binding(2) var<storage, read_write> out: array<Field>;
@group(0) @binding(3) var<storage, read_write> pih: array<Field>;
@group(0) @binding(4) var<storage, read_write> unused: array<Field>;
fn wire(row: u32, i: u32) -> Field {{ return src[row * tp.stride + i]; }}
fn constant(row: u32, j: u32) -> Field {{ return src[row * tp.stride + tp.num_wires + j]; }}
fn public_input(k: u32) -> Field {{ return pih[k]; }}
{}
@compute @workgroup_size(64)
fn gate_test(@builtin(global_invocation_id) id: vec3<u32>) {{
    let row = id.x;
    if (row >= tp.count) {{ return; }}
    var cs: array<Field, {nc}>;
    gate_0(row, Field(1u, 0u), &cs);
    for (var k = 0u; k < {nc}u; k++) {{ out[row * {nc}u + k] = canonical(cs[k]); }}
    unused[0] = Field(0u, 0u);
}}
"#,
            trace::emit_gate(&trace, "gate_0", num_constraints),
            nc = num_constraints
        ));
        if let Ok(nonce) = std::env::var("PLONKY2_GPU_KERNEL_NONCE") {
            source.push_str(&format!("\n// nonce {nonce}\n"));
        }
        let compile_start = std::time::Instant::now();
        let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = module(&ctx.device, "gate test", &source);
        let layout = storage_layout(
            &ctx.device,
            "gate test",
            &[
                (true, true),
                (false, true),
                (false, false),
                (false, false),
                (false, false),
            ],
        );
        let pipeline = pipeline(&ctx.device, &layout, &module, "gate_test");
        if let Some(error) = pollster::block_on(validation.pop()) {
            panic!("{}: {error}", gate.0.id());
        }
        eprintln!(
            "{}: {} nodes, kernel compiled in {:?}",
            gate.0.id(),
            trace.dag.nodes.len(),
            compile_start.elapsed()
        );
        let mut words = Vec::with_capacity(batch * stride);
        for row in 0..batch {
            words.extend((0..num_wires).map(|i| wires[i][row].to_noncanonical_u64()));
            words.extend((0..num_constants).map(|j| constants[j][row].to_noncanonical_u64()));
        }
        let src = ctx.upload("gate inputs", &words);
        let out = ctx.storage("gate outputs", (batch * num_constraints * 8) as u64);
        let pih_buffer = ctx.upload("pih", &field_words(&pih.elements));
        let unused = ctx.storage("unused", 16);
        let params = ctx.uniform(
            "gate params",
            &[batch as u32, stride as u32, num_wires as u32, 0],
        );
        let bindings = ctx.bind(&layout, &[&params, &src, &out, &pih_buffer, &unused]);
        let mut encoder = ctx.encoder("gate test");
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((batch as u32).div_ceil(64), 1, 1);
        }
        ctx.submit(encoder);
        let actual = ctx.read_u64(&out, batch * num_constraints);
        for row in 0..batch {
            for k in 0..num_constraints {
                assert_eq!(
                    F::from_canonical_u64(actual[row * num_constraints + k]),
                    expected[k * batch + row],
                    "{} row {row} constraint {k}",
                    gate.0.id()
                );
            }
        }
    }
}

#[test]
fn hybrid_proof_plan_keeps_leaves_and_prepares_quotient_rows() {
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let (data, pw) = hash_chain_circuit(256);
    let plan = policy::ExecutionPlan::new(&data.common, &data.prover_only);
    if plan.path != policy::ExecutionPath::Hybrid {
        return; // Discrete adapters default to resident proofs.
    }
    // Within a proof, hybrid keeps leaves on unified memory; outside it does not.
    assert_eq!(plan.retain_leaves(), ctx.mappable);
    assert!(!ctx.options.retain_leaves(ctx.mappable));
    let degree = data.common.degree();
    let rate_bits = data.common.config.fri_config.rate_bits;
    let values = (0..5)
        .map(|c| {
            PolynomialValues::new(
                (0..degree)
                    .map(|i| F::from_canonical_usize(7 * i + c))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let commit = |plan| {
        PolynomialBatch::<F, C, 2>::from_values_with_plan(
            values.clone(),
            rate_bits,
            false,
            4,
            &mut TimingTree::default(),
            None,
            plan,
            false,
        )
    };
    let in_proof = commit(Some(&plan));
    let outside = commit(None);
    assert!(!outside.merkle_tree.leaves.is_device_resident());
    assert_eq!(in_proof.merkle_tree.cap, outside.merkle_tree.cap);
    let step = 2;
    let points = 100..300;
    in_proof.prepare_lde_values(points.clone(), step);
    for i in points {
        assert_eq!(
            in_proof.get_lde_values(i, step),
            outside.get_lde_values(i, step)
        );
    }
    // The prepared rows came from the mapped buffer, not a full download.
    assert_eq!(
        in_proof.merkle_tree.leaves.is_device_resident(),
        ctx.mappable
    );
    drop((in_proof, outside, plan));
    data.verify(data.prove(pw).unwrap()).unwrap();
}

#[test]
fn proof_commitments_prepare_query_paths_without_digest_download() {
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let (data, pw) = hash_chain_circuit(256);
    let plan = policy::ExecutionPlan::new(&data.common, &data.prover_only);
    if !plan.retain_leaves() {
        return;
    }
    let degree = data.common.degree();
    let rate_bits = data.common.config.fri_config.rate_bits;
    let values = (0..5)
        .map(|c| {
            PolynomialValues::new(
                (0..degree)
                    .map(|i| F::from_canonical_usize(3 * i + c))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let commit = |plan| {
        PolynomialBatch::<F, C, 2>::from_values_with_plan(
            values.clone(),
            rate_bits,
            false,
            4,
            &mut TimingTree::default(),
            None,
            plan,
            false,
        )
    };
    let in_proof = commit(Some(&plan));
    let outside = commit(None);
    let tree = &in_proof.merkle_tree;
    assert!(tree.digests.is_device_resident());
    // Outside a proof the context's own policy decides: resident mode (the
    // default on discrete adapters) keeps digests on the device as it keeps leaves.
    assert_eq!(
        outside.merkle_tree.digests.is_device_resident(),
        ctx.options.retain_leaves(ctx.mappable)
    );
    assert_eq!(tree.cap, outside.merkle_tree.cap);
    let leaves = tree.leaves.len();
    let indices = [0, 7, leaves - 1, 7, leaves / 2];
    tree.digests
        .prepare_paths(&indices, leaves, crate::util::log2_strict(tree.cap.len()));
    for i in indices {
        assert_eq!(tree.prove(i), outside.merkle_tree.prove(i));
    }
    assert!(tree.digests.is_device_resident());
    // Any other path falls back to the full download, with the same result.
    assert_eq!(tree.prove(3), outside.merkle_tree.prove(3));
    assert!(!tree.digests.is_device_resident());
    assert_eq!(tree.digests, outside.merkle_tree.digests);
    drop((in_proof, outside, plan));
    data.verify(data.prove(pw).unwrap()).unwrap();
}

/// A host tree and the same tree with its digests held on the device.
fn device_digest_tree(
    ctx: &Arc<GpuContext>,
    log_n: usize,
    cap_height: usize,
) -> (
    MerkleTree<F, crate::hash::poseidon::PoseidonHash>,
    MerkleTree<F, crate::hash::poseidon::PoseidonHash>,
) {
    use crate::hash::merkle_tree::MerkleDigests;
    use crate::hash::poseidon::PoseidonHash as H;
    use crate::plonk::config::GenericHashOut;
    let (n, leaf_len) = (1usize << log_n, 4);
    let flat = (0..n * leaf_len)
        .map(|i| F::from_canonical_u64((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 8))
        .collect::<Vec<_>>();
    let host = MerkleTree::<F, H>::new_cpu(MerkleLeaves::new(flat, leaf_len, n), cap_height);
    let words = host
        .digests
        .iter()
        .flat_map(|h| {
            h.to_bytes()
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let buffer = ctx.upload("test digests", &words);
    let done = ctx.submit_async(ctx.encoder("test digests ready"));
    let digests = MerkleDigests::from_device::<F, H>(ctx.clone(), buffer, host.digests.len(), done);
    let device = MerkleTree {
        leaves: host.leaves.clone(),
        digests,
        cap: host.cap.clone(),
    };
    (host, device)
}

#[test]
fn device_digests_replace_path_sets_and_fall_back_inside_rayon() {
    use plonky2_maybe_rayon::*;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let log_n = 16;
    let n = 1 << log_n;
    let (host, device) = device_digest_tree(&ctx, log_n, 4);
    // A later query set replaces the earlier one without a download.
    device.digests.prepare_paths(&[0, 1, 2], n, 4);
    device.digests.prepare_paths(&[5, n - 1], n, 4);
    for i in [5, n - 1] {
        assert_eq!(device.prove(i), host.prove(i));
    }
    assert!(device.digests.is_device_resident());
    // Unprepared paths from a parallel loop take the full download once; the
    // decode must not run on the Rayon pool while it holds the lock.
    let device = Arc::new(device);
    let (send, recv) = std::sync::mpsc::channel();
    let worker = device.clone();
    std::thread::spawn(move || {
        let proofs = (0..4096usize)
            .into_par_iter()
            .map(|i| worker.prove((i * 7919) % n))
            .collect::<Vec<_>>();
        let _ = send.send(proofs);
    });
    let proofs = recv
        .recv_timeout(std::time::Duration::from_secs(120))
        .expect("parallel fallback deadlocked");
    for (k, proof) in proofs.iter().enumerate() {
        assert_eq!(*proof, host.prove((k * 7919) % n));
    }
    assert!(!device.digests.is_device_resident());
}

#[test]
fn prepared_leaf_rows_match_without_full_download() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let leaf_len = 13;
    let num_leaves = 1024;
    let words = (0..leaf_len * num_leaves)
        .map(|i| (i as u64).wrapping_mul(0x0807060504030201) ^ u64::MAX)
        .collect::<Vec<_>>();
    let expected = words
        .iter()
        .map(|&w| F::from_noncanonical_u64(w))
        .collect::<Vec<_>>();
    let row = |i: usize| &expected[i * leaf_len..(i + 1) * leaf_len];
    let buffer = ctx.upload("prepared test leaves", &words);
    let done = ctx.submit_async(ctx.encoder("prepared test ready"));
    let leaves = MerkleLeaves::<F>::from_device(ctx.clone(), buffer, leaf_len, num_leaves, done);
    leaves.prepare_rows(&[700, 3, 1023, 3]);
    if ctx.mappable {
        // Prepared rows and query rows coexist without downloading everything.
        assert!(leaves.is_device_resident());
        for i in [3, 700, 1023] {
            assert_eq!(leaves.row(i), row(i));
        }
        leaves.prepare_queries(&[5, 700]);
        assert_eq!(leaves.row(5), row(5));
        assert_eq!(leaves.row(700), row(700));
        assert!(leaves.is_device_resident());
        // Only the first set is kept; any other row takes the full download.
        leaves.prepare_rows(&[9]);
        assert!(leaves.is_device_resident());
    }
    assert_eq!(leaves.row(9), row(9));
    assert_eq!(leaves.row(3), row(3));
    assert_eq!(leaves.as_flat_slice(), &expected);
    assert!(!leaves.is_device_resident());
}

#[test]
fn resident_leaf_readback_matches_full_rows() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let leaf_len = 13;
    let num_leaves = 1024;
    let words = (0..leaf_len * num_leaves)
        .map(|i| (i as u64).wrapping_mul(0x102030405060708))
        .collect::<Vec<_>>();
    let buffer = ctx.upload("resident test leaves", &words);
    let done = ctx.submit_async(ctx.encoder("resident test ready"));
    let leaves = MerkleLeaves::<F>::from_device(ctx.clone(), buffer, leaf_len, num_leaves, done);
    let mut clone = leaves.clone();
    assert_eq!(leaves.num_values(), words.len());
    assert_eq!(leaves.len(), num_leaves);
    assert!(!leaves.is_empty());
    assert!(leaves.is_device_resident()); // Metadata must not download.
    let indices = [0, 1, 511, 1023, 1];
    leaves.prepare_queries(&indices);
    for i in indices {
        let expected = words[i * leaf_len..(i + 1) * leaf_len]
            .iter()
            .map(|&w| F::from_noncanonical_u64(w))
            .collect::<Vec<_>>();
        assert_eq!(leaves.row(i), expected);
    }
    // An unprepared row triggers the full CPU fallback; all data must remain exact.
    let expected = words
        .iter()
        .map(|&w| F::from_noncanonical_u64(w))
        .collect::<Vec<_>>();
    assert_eq!(leaves.row(9), &expected[9 * leaf_len..10 * leaf_len]);
    assert_eq!(leaves.as_flat_slice(), &expected);
    assert!(!leaves.is_device_resident());
    assert!(!clone.is_device_resident()); // Shared buffer released for every clone.
    clone.as_flat_slice_mut()[0] = F::ONE;
    assert_eq!(clone.row(0)[0], F::ONE);
    assert_eq!(leaves.row(0)[0], expected[0]);
    assert_eq!(leaves.clone().into_flat_vec(), expected);
    let buffer = ctx.upload("resident serialization leaves", &words);
    let done = ctx.submit_async(ctx.encoder("resident serialization ready"));
    let fresh = MerkleLeaves::<F>::from_device(ctx.clone(), buffer, leaf_len, num_leaves, done);
    let host = MerkleLeaves::new(expected, leaf_len, num_leaves);
    let encoded = serde_json::to_value(&fresh).unwrap();
    assert!(!fresh.is_device_resident());
    assert_eq!(
        encoded,
        serde_json::json!({
            "values": host.as_flat_slice(), "leaf_len": leaf_len, "num_leaves": num_leaves
        })
    );
    assert_eq!(encoded, serde_json::to_value(&host).unwrap());
    assert_eq!(
        serde_cbor::to_vec(&fresh).unwrap(),
        serde_cbor::to_vec(&host).unwrap()
    );
    let decoded: MerkleLeaves<F> = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, host);
}

#[test]
fn opening_fft_matches_cpu_and_skips_small_inputs() {
    use crate::field::extension::quadratic::QuadraticExtension;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(_ctx) = ctx() else { return };
    let floor = fri::OPENING_FFT_MIN_LOG_N;
    for log_n in [0, floor - 1, floor, floor + 1] {
        let coeffs = PolynomialCoeffs::new(
            (0..1 << log_n)
                .map(|i| QuadraticExtension([F::from_canonical_usize(i), F::ONE]))
                .collect(),
        );
        let actual = fri::try_opening_fft::<F, 2>(&coeffs);
        if log_n < floor {
            assert!(actual.is_none());
        } else {
            assert_eq!(
                actual.unwrap().values,
                coeffs.coset_fft(F::coset_shift().into()).values
            );
        }
    }
}

#[test]
fn discrete_memory_policy_and_readbacks() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(_) = ctx() else { return };
    // Exercise the staging and resident paths even on an integrated Mac GPU.
    // This owned context is explicitly dropped; it is not the global context.
    let mut ctx = GpuContext::new().unwrap();
    ctx.mappable = false;
    ctx.options.mode = policy::ExecutionMode::Auto;
    assert!(ctx.options.retain_leaves(false));
    assert!(!ctx.options.retain_leaves(true));
    ctx.options.mode = policy::ExecutionMode::Hybrid;
    assert!(!ctx.options.retain_leaves(false));
    ctx.options.mode = policy::ExecutionMode::Auto;
    let ctx = Arc::new(ctx);
    let words = (0..1 << 18).map(|i| i as u64).collect::<Vec<_>>();
    let buffer = ctx.upload("staging test", &words);
    for _ in 0..3 {
        assert_eq!(ctx.read_u64(&buffer, words.len()), words);
        assert_eq!(ctx.readback_pool.lock().unwrap().len(), 1);
    }
    let done = ctx.submit_async(ctx.encoder("resident staging ready"));
    let leaves = MerkleLeaves::<F>::from_device(ctx.clone(), buffer, 16, words.len() / 16, done);
    leaves.prepare_queries(&[0, 7, 19]);
    assert!(leaves.is_device_resident());
    // Concurrent full and sparse readers also exercise the parallel decoder.
    std::thread::scope(|scope| {
        for i in 0..8 {
            let leaves = &leaves;
            scope.spawn(move || {
                leaves.prepare_queries(&[19, 500]);
                let expected = words_for_row(i * 10, 16);
                assert_eq!(leaves.row(i * 10), expected);
            });
        }
    });
    assert!(!leaves.is_device_resident());
    assert_eq!(Arc::strong_count(&ctx), 1);
    assert_eq!(
        leaves.into_flat_vec(),
        words
            .iter()
            .map(|&w| F::from_noncanonical_u64(w))
            .collect::<Vec<_>>()
    );

    fn words_for_row(row: usize, len: usize) -> Vec<F> {
        (row * len..(row + 1) * len)
            .map(F::from_canonical_usize)
            .collect()
    }
}

/// Run in its own process: successful shutdown permanently disables the GPU.
#[test]
#[ignore = "requires an isolated process and a GPU"]
fn discrete_memory_proof_and_shutdown() {
    memory_policy_proof_and_shutdown(false);
}

/// Uses real shared-memory mapping rather than emulating a discrete GPU.
#[test]
#[ignore = "requires an isolated process and a unified-memory GPU"]
fn unified_resident_proof_and_shutdown() {
    memory_policy_proof_and_shutdown(true);
}

fn memory_policy_proof_and_shutdown(unified: bool) {
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    let _ = env_logger::try_init();
    assert!(matches!(
        *CONTEXT.lock().unwrap(),
        ContextState::Uninitialized
    ));
    let mut ctx = GpuContext::new().expect("test requires a GPU");
    if unified {
        assert!(ctx.mappable, "test requires a unified-memory GPU");
        ctx.options.mode = policy::ExecutionMode::Resident;
    } else {
        ctx.mappable = false;
        ctx.options.mode = policy::ExecutionMode::Auto;
    }
    let expect_resident = ctx.options.retain_leaves(ctx.mappable);
    *CONTEXT.lock().unwrap() = ContextState::Available(Some(Arc::new(ctx)));
    {
        let ctx = context().unwrap();
        let values = vec![PolynomialValues::new(vec![F::ONE; 1024]); 3];
        for blinding in [false, true] {
            let batch = lde::from_values::<F, C, 2>(
                &ctx,
                values.clone(),
                2,
                blinding,
                4,
                &mut TimingTree::default(),
            );
            assert_eq!(
                batch.merkle_tree.leaves.is_device_resident(),
                expect_resident
            );
        }
    }
    let (data, pw) = hash_chain_circuit(256);
    let initial_share = *context().unwrap().quotient_cpu_share.lock().unwrap();
    {
        let prepared = quotient::prepare(&data.common, &data.prover_only);
        assert!(prepared.is_some());
        let hybrid = policy::ExecutionPlan::from_prepared(
            prepared.clone(),
            Some(policy::ExecutionMode::Hybrid),
        );
        let resident =
            policy::ExecutionPlan::from_prepared(prepared, Some(policy::ExecutionMode::Resident));
        assert_eq!(hybrid.path, policy::ExecutionPath::Hybrid);
        assert_eq!(resident.path, policy::ExecutionPath::Resident);
        assert_eq!(resident.cpu_share, 0.0);
        let ctx = context().unwrap();
        *ctx.quotient_cpu_share.lock().unwrap() = 0.33;
        assert_eq!(hybrid.cpu_share, initial_share);
        *ctx.quotient_cpu_share.lock().unwrap() = initial_share;
    }
    for _ in 0..3 {
        data.verify(data.prove(pw.clone()).unwrap()).unwrap();
    }
    if expect_resident {
        // The automatic GPU-only choice must not feed bogus timings into the
        // adaptive CPU share used for host-backed circuits.
        assert_eq!(
            *context().unwrap().quotient_cpu_share.lock().unwrap(),
            initial_share
        );
    }
    // Unsalted leaves must support the same sparse queries as blinded ones.
    // Build a separate CPU reference and reuse the GPU prover across A-B-A inputs.
    {
        use crate::iop::witness::{PartialWitness, WitnessWrite};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::Hasher;
        let build = || {
            let mut builder =
                CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
            let input = builder.add_virtual_target();
            builder.register_public_input(input);
            let three = builder.constant(F::from_canonical_u64(3));
            let seven = builder.constant(F::from_canonical_u64(7));
            let mut state = vec![input; 4];
            for _ in 0..64 {
                for x in &mut state {
                    *x = builder.mul_add(*x, three, seven);
                }
                state = builder.hash_n_to_hash_no_pad::<H>(state).elements.to_vec();
            }
            builder.register_public_inputs(&state);
            while builder.num_gates() < 960 {
                builder.add_gate(crate::gates::noop::NoopGate, vec![]);
            }
            (builder.build::<C>(), input)
        };
        set_enabled(false);
        let (cpu, input) = build();
        set_enabled(true);
        let (gpu, gpu_input) = build();
        assert_eq!(input, gpu_input);
        assert_eq!(gpu.common, cpu.common);
        assert_eq!(gpu.verifier_only, cpu.verifier_only);
        assert!(!gpu.common.config.zero_knowledge);
        let plan = policy::ExecutionPlan::new(&gpu.common, &gpu.prover_only);
        assert_eq!(plan.path, policy::ExecutionPath::Resident);
        assert_eq!(plan.cpu_share, 0.0);
        for value in [7, 19, 7, 23] {
            let value = F::from_canonical_u64(value);
            let mut pw = PartialWitness::new();
            pw.set_target(input, value).unwrap();
            let proof = gpu.prove(pw).unwrap();
            let mut state = vec![value; 4];
            for _ in 0..64 {
                for x in &mut state {
                    *x = *x * F::from_canonical_u64(3) + F::from_canonical_u64(7);
                }
                state = H::hash_no_pad(&state).elements.to_vec();
            }
            assert_eq!(proof.public_inputs, [vec![value], state].concat());
            cpu.verify(proof).unwrap();
        }
    }
    // Lookups are unsupported by the GPU quotient kernel. The plan must choose
    // host commitments up front, even when resident execution is requested.
    {
        use crate::gates::noop::NoopGate;
        use crate::iop::witness::{PartialWitness, WitnessWrite};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        let mut builder =
            CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_zk_config());
        let input = builder.add_virtual_target();
        let table = builder.add_lookup_table_from_fn(|x| x * x, &[0, 1, 2, 3, 4, 5, 6, 7]);
        let output = builder.add_lookup_from_index(input, table);
        builder.register_public_input(output);
        for _ in 0..256 {
            builder.add_gate(NoopGate, vec![]);
        }
        let lookup_data = builder.build::<C>();
        assert!(lookup_data.common.num_lookup_polys > 0);
        let mut pw = PartialWitness::new();
        pw.set_target(input, F::from_canonical_u64(7)).unwrap();
        let plan = policy::ExecutionPlan::new(&lookup_data.common, &lookup_data.prover_only);
        assert_eq!(plan.path, policy::ExecutionPath::CpuQuotient);
        assert!(!plan.retain_leaves());
        let batch = PolynomialBatch::<F, C, 2>::from_values_with_plan(
            vec![PolynomialValues::new(vec![
                F::ONE;
                lookup_data.common.degree()
            ])],
            lookup_data.common.config.fri_config.rate_bits,
            true,
            4,
            &mut TimingTree::default(),
            None,
            Some(&plan),
            false,
        );
        assert!(!batch.merkle_tree.leaves.is_device_resident());
        drop(batch);
        // Different immutable paths coexist on one context. Resident proofs
        // must not materialize just because the lookup proof needs host leaves.
        std::thread::scope(|scope| {
            let hash = scope.spawn(|| {
                let (data, pw) = hash_chain_circuit(256);
                let plan = policy::ExecutionPlan::new(&data.common, &data.prover_only);
                assert_eq!(plan.retain_leaves(), expect_resident);
                drop(plan);
                for _ in 0..3 {
                    data.verify(data.prove(pw.clone()).unwrap()).unwrap();
                }
            });
            let lookup = scope.spawn(|| {
                for _ in 0..3 {
                    let proof = lookup_data.prove(pw.clone()).unwrap();
                    assert_eq!(proof.public_inputs, vec![F::from_canonical_u64(49)]);
                    lookup_data.verify(proof).unwrap();
                }
            });
            hash.join().unwrap();
            lookup.join().unwrap();
        });
    }
    shutdown().unwrap();
    data.verify(data.prove(pw).unwrap()).unwrap();
}

/// Run in its own process: successful shutdown permanently disables the GPU.
#[test]
#[ignore = "requires an isolated process and a GPU"]
fn resident_cached_setup_shutdown() {
    let _ = env_logger::try_init();
    let mut ctx = GpuContext::new().expect("test requires a GPU");
    ctx.options.mode = policy::ExecutionMode::Resident;
    *CONTEXT.lock().unwrap() = ContextState::Available(Some(Arc::new(ctx)));
    // Application caches can retain circuits that have never proved anything.
    let (unused, _) = hash_chain_circuit(256);
    let cloned = unused.clone();
    assert!(
        !unused
            .prover_only
            .constants_sigmas_commitment
            .merkle_tree
            .leaves
            .is_device_resident(),
        "fixed setup leaves must not pin the GPU context"
    );
    let (data, pw) = hash_chain_circuit(256);
    data.verify(data.prove(pw.clone()).unwrap()).unwrap();
    shutdown().expect("cached setup must permit orderly shutdown");
    assert!(!is_available());
    assert!(unused
        .prover_only
        .constants_sigmas_commitment
        .device
        .get()
        .is_none());
    assert!(cloned
        .prover_only
        .constants_sigmas_commitment
        .device
        .get()
        .is_none());
    data.verify(data.prove(pw).unwrap()).unwrap();
}

/// Run in its own process: successful shutdown permanently disables the GPU.
#[test]
#[ignore = "requires an isolated process and a GPU"]
fn orderly_shutdown() {
    let _ = env_logger::try_init();
    let ctx = context().expect("shutdown test requires a GPU");
    let retained = ctx.retained.clone();
    assert!(matches!(shutdown(), Err(ShutdownError::InUse)));
    let weak = Arc::downgrade(&ctx);
    drop(ctx);
    assert!(matches!(shutdown(), Err(ShutdownError::InUse)));
    drop(weak);
    let ctx = context().unwrap();
    std::thread::spawn(move || {
        assert!(matches!(shutdown(), Err(ShutdownError::InUse)));
        drop(ctx);
    })
    .join()
    .unwrap();

    // Exercise a real proof, leaving the circuit's retained LDE alive.
    let (data, pw) = hash_chain_circuit(256);
    data.verify(data.prove(pw.clone()).unwrap()).unwrap();
    let cached = &data.prover_only.constants_sigmas_commitment.device;
    assert!(cached.get().is_some());
    let cloned = cached.clone();
    assert!(cloned.get().is_some());
    assert!(retained.bytes() > 0);

    // Submit work without a readback: shutdown itself must wait for it.
    let ctx = context().unwrap();
    let buffer = ctx.storage("shutdown pending work", 8);
    let mut encoder = ctx.encoder("shutdown pending work");
    encoder.clear_buffer(&buffer, 0, None);
    ctx.submit_async(encoder);
    drop(buffer);
    drop(ctx);
    // A resident leaf clone must block shutdown until either dropped or fully
    // materialized; after materialization both clones remain usable on CPU.
    let ctx = context().unwrap();
    let buffer = ctx.upload("shutdown resident leaves", &[3, 4, 5, 6]);
    let done = ctx.submit_async(ctx.encoder("shutdown resident leaves ready"));
    let leaves = MerkleLeaves::<F>::from_device(ctx.clone(), buffer, 2, 2, done);
    let leaf_clone = leaves.clone();
    drop(ctx);
    assert!(matches!(shutdown(), Err(ShutdownError::InUse)));
    assert_eq!(leaves.as_flat_slice()[0], F::from_canonical_u64(3));
    assert!(!leaf_clone.is_device_resident());
    shutdown().unwrap();
    assert_eq!(
        leaf_clone.row(1),
        [F::from_canonical_u64(5), F::from_canonical_u64(6)]
    );
    assert!(cached.get().is_none());
    assert!(cloned.get().is_none());
    assert_eq!(retained.bytes(), 0);
    assert!(!is_available());
    set_enabled(true);
    assert!(!is_available());
    shutdown().unwrap();
    // The same circuit remains usable on the CPU after device buffers clear.
    data.verify(data.prove(pw).unwrap()).unwrap();
    println!("orderly GPU shutdown and CPU fallback passed");
}

#[test]
#[ignore = "requires an isolated process"]
fn shutdown_before_initialization() {
    assert!(matches!(
        *CONTEXT.lock().unwrap(),
        ContextState::Uninitialized
    ));
    shutdown().unwrap();
    shutdown().unwrap();
    set_enabled(true);
    assert!(context().is_none());
}

#[cfg(feature = "gpu-profiling")]
mod benchmarks;

/// The `Vec<Vec<F>>` view of Merkle leaves exists for code written against
/// upstream plonky2; proofs read rows directly and must never build it, or
/// the device-resident leaves would be downloaded and copied in full.
#[test]
fn proving_never_builds_the_compatible_row_view() {
    use core::sync::atomic::Ordering;

    use crate::hash::merkle_tree::DEVICE_ROW_VIEWS;
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = ctx() else { return };
    // Control: dereferencing device-backed leaves builds the view and counts.
    let words = (0..4 * 1024u64).collect::<Vec<_>>();
    let buffer = ctx.upload("row view control leaves", &words);
    let done = ctx.submit_async(ctx.encoder("row view control ready"));
    let leaves = MerkleLeaves::<F>::from_device(ctx.clone(), buffer, 4, 1024, done);
    let before = DEVICE_ROW_VIEWS.load(Ordering::Relaxed);
    assert_eq!(leaves[3], [12, 13, 14, 15].map(F::from_canonical_u64));
    assert_eq!(DEVICE_ROW_VIEWS.load(Ordering::Relaxed), before + 1);
    // Proofs with the worklist and with the compiled executor.
    let rows = 1usize << crate::gpu::MIN_LOG_LEAVES.saturating_sub(2).max(7);
    let (data, pw) = hash_chain_circuit(rows);
    let before = DEVICE_ROW_VIEWS.load(Ordering::Relaxed);
    for executor in [false, true] {
        if executor {
            let program = data.compile_witness_program_like(&pw).unwrap();
            data.install_witness_program(program).unwrap();
        }
        let proof = data.prove(pw.clone()).unwrap();
        data.verify(proof).unwrap();
    }
    assert_eq!(DEVICE_ROW_VIEWS.load(Ordering::Relaxed), before);
}

/// Run in its own process: the exit-time shutdown skips while the GPU is in
/// use or its lock is held, without blocking, and otherwise shuts down.
#[test]
#[ignore = "requires an isolated process and a GPU"]
fn exit_shutdown_skips_while_in_use() {
    let _ = env_logger::try_init();
    let ctx = context().expect("exit shutdown test requires a GPU");
    let retained = ctx.retained.clone();
    shutdown_at_exit();
    assert!(is_available(), "a held handle must keep the GPU");
    drop(ctx);
    // Another thread holding the context lock: return at once, not wait.
    let (locked, release) = (std::sync::Barrier::new(2), std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let _state = CONTEXT.lock().unwrap();
            locked.wait();
            release.wait();
        });
        locked.wait();
        let started = std::time::Instant::now();
        shutdown_at_exit();
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        release.wait();
    });
    assert!(is_available());
    // A proof leaves a retained LDE; with no handle held, exit shuts down.
    let (data, pw) = hash_chain_circuit(256);
    data.verify(data.prove(pw.clone()).unwrap()).unwrap();
    assert!(retained.bytes() > 0);
    shutdown_at_exit();
    assert!(!is_available());
    assert_eq!(retained.bytes(), 0);
    data.verify(data.prove(pw).unwrap()).unwrap();
}

/// Child of `exit_shutdown_runs_at_process_exit`: proves and returns without
/// calling `shutdown`, leaving it to the exit handler.
#[test]
#[ignore = "run by exit_shutdown_runs_at_process_exit"]
fn exit_shutdown_child() {
    let _ = env_logger::builder().is_test(false).try_init();
    let (data, pw) = hash_chain_circuit(256);
    data.verify(data.prove(pw).unwrap()).unwrap();
    assert!(is_available());
}

/// Run in its own process: an application that never calls `shutdown` still
/// releases the GPU at normal exit, and exits cleanly.
#[test]
#[ignore = "requires an isolated process and a GPU"]
fn exit_shutdown_runs_at_process_exit() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "gpu::tests::exit_shutdown_child",
            "--test-threads=1",
        ])
        .env("RUST_LOG", "plonky2::gpu=debug")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "child failed: {stderr}");
    assert!(
        stderr.contains("plonky2 gpu: shut down at process exit"),
        "no exit shutdown in: {stderr}"
    );
}

#[test]
fn subgroup_values_are_retained_only_for_permutation() {
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    let config = GpuConfig {
        allow_software: true,
        ..GpuConfig::default()
    };
    let Ok(ctx) = GpuContext::with_config(&config) else {
        eprintln!("no compute adapter; skipping");
        return;
    };
    let ctx = Arc::new(ctx);
    for retain_values in [false, true] {
        let batch = lde::from_values_with_residency::<F, C, 2>(
            &ctx,
            vec![PolynomialValues::new(vec![F::ONE; 256])],
            2,
            false,
            2,
            &mut TimingTree::default(),
            false,
            retain_values,
        );
        assert_eq!(batch.device_values.get().is_some(), retain_values);
        assert_eq!(
            ctx.retained.bytes(),
            0,
            "proof buffers are not persistent caches"
        );
        assert!(batch.polynomials[0].coeffs[0] == F::ONE);
    }
}

/// Isolated because it configures the process-wide context to accept a software adapter.
#[test]
#[ignore = "run alone with --ignored to exercise the complete Vulkan prover on software"]
fn software_gpu_proofs_verify() {
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    configure(GpuConfig {
        backend: Some(Backend::Vulkan),
        allow_software: true,
        ..Default::default()
    })
    .unwrap();
    assert!(context().is_some(), "a Vulkan adapter is required");
    let mut config = CircuitConfig::standard_recursion_config();
    config.security_bits = 1;
    config.fri_config.num_query_rounds = 2;
    config.fri_config.proof_of_work_bits = 0;
    let mut builder = CircuitBuilder::<F, 2>::new(config);
    let input = builder.add_virtual_target();
    let mut state = vec![input; 4];
    for _ in 0..128 {
        state = builder.hash_n_to_hash_no_pad::<H>(state).elements.to_vec();
    }
    builder.register_public_inputs(&state);
    let data = builder.build::<C>();
    let plan = policy::ExecutionPlan::new(&data.common, &data.prover_only);
    assert_eq!(plan.path, policy::ExecutionPath::Resident);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u64(42)).unwrap();
    data.verify(data.prove(pw.clone()).unwrap()).unwrap();
    let program = data.compile_witness_program_like(&pw).unwrap();
    data.install_witness_program(program).unwrap();
    data.verify(data.prove(pw).unwrap()).unwrap();
}

#[test]
fn quotient_and_openings_share_the_fixed_coefficient_upload() {
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    let config = GpuConfig {
        allow_software: true,
        ..Default::default()
    };
    let Ok(ctx) = GpuContext::with_config(&config) else {
        eprintln!("no compute adapter; skipping");
        return;
    };
    let batch = PolynomialBatch::<F, C, 2>::from_coeffs_cpu(
        vec![PolynomialCoeffs::new(vec![F::ONE; 256])],
        2,
        false,
        2,
        &mut TimingTree::default(),
        None,
    );
    let first = lde::device_coefficients(&ctx, &batch, "quotient_lde");
    let bytes = ctx.retained.bytes();
    assert_eq!(bytes, first.size());
    let cloned = batch.clone();
    assert_eq!(
        ctx.retained.bytes(),
        bytes,
        "clones share the allocation reservation"
    );
    let second = lde::device_coefficients(&ctx, &cloned, "opening");
    assert_eq!(*first, *second);
    drop(batch);
    assert_eq!(ctx.retained.bytes(), bytes);
    drop(cloned);
    assert_eq!(ctx.retained.bytes(), 0);
}
