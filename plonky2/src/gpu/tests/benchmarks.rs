//! Optional GPU profiling probes; run individually with `gpu-profiling`.

use super::*;

/// Alternate warmed tile plans on identical buffers. Run alone so timings
/// include no concurrent proofs or shader compilation from other tests.
#[test]
#[ignore]
fn bench_fft_tile_plans() {
    use std::time::Instant;

    use crate::gpu::lde::{fft_groups_with_tile, FftJob};
    let ctx = ctx().expect("benchmark requires a GPU");
    assert!(ctx.device.limits().max_compute_workgroup_storage_size >= 16384);
    let source = shader::fft_module_with_tile(ctx.flavor, 11)
        .replace("@workgroup_size(256)", "@workgroup_size(512)")
        .replace("+= 256u", "+= 512u");
    let module = module(&ctx.device, "fft tile 2048 workgroup 512", &source);
    let large_512 = pipeline(&ctx.device, &ctx.fft_layout, &module, "fft_group");
    let mut rng = StdRng::seed_from_u64(0x54494c45);
    for (log_n, columns) in [
        (16u32, 135usize),
        (17, 135),
        (18, 135),
        (19, 135),
        (19, 20),
        (19, 2),
    ] {
        let n = 1usize << log_n;
        let degree = n / 8;
        let input = random_columns(&mut rng, columns, degree);
        let slices = input.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let input = lde::upload_columns(&ctx, "fft plan input", &slices);
        let data = ctx.storage("fft plan data", (columns * n * 8) as u64);
        let run = |variant| {
            let bits = if variant == 0 { 10 } else { 11 };
            let mut encoder = ctx.encoder("fft plan benchmark");
            fft_groups_with_tile(
                &ctx,
                &mut encoder,
                FftJob {
                    log_n,
                    columns: columns as u32,
                    input: &input,
                    in_len: degree as u32,
                    in_stride: degree as u32,
                    data: &data,
                    output: None,
                    shift_in: Some(F::coset_shift()),
                    inverse: false,
                    shift_out: None,
                },
                bits,
                None,
                (variant == 2).then_some(&large_512),
            );
            let start = Instant::now();
            ctx.submit(encoder);
            start.elapsed().as_secs_f64() * 1000.0
        };
        for variant in 0..3 {
            for _ in 0..3 {
                run(variant);
            }
        }
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..12 {
            for i in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                samples[i].push(run(i));
            }
        }
        for (i, sample) in samples.iter_mut().enumerate() {
            sample.sort_by(f64::total_cmp);
            println!("FFT_TILE log_n={log_n} columns={columns} bits={} workgroup={} median_ms={:.6} samples={sample:?}", if i == 0 {10} else {11}, if i == 2 {512} else {256}, (sample[5] + sample[6]) / 2.0);
        }
    }
}

/// Full digest readback versus caps and 28 gathered authentication paths.
/// Paths are checked outside timing against the full tree on every iteration.
#[test]
#[ignore]
fn bench_merkle_query_readback() {
    use std::time::Instant;
    let _ = env_logger::try_init();
    let ctx = context().expect("benchmark requires a GPU");
    assert!(
        ctx.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY),
        "set PLONKY2_GPU_TIMESTAMPS=1 on a supported GPU"
    );
    let log_n = 19u32;
    let n = 1usize << log_n;
    let cap_height = 4;
    let columns = 139;
    let source = ctx.storage("query benchmark source", (n * columns * 8) as u64);
    let mut init = ctx.encoder("initialize query benchmark source");
    init.clear_buffer(&source, 0, None);
    ctx.submit(init);
    let indices = (0..28).map(|i| (i * 18937 + 17) % n).collect::<Vec<_>>();
    for iteration in 0..16 {
        let tree = merkle::build_tree(
            &ctx,
            crate::plonk::config::GpuHash::Poseidon,
            &source,
            merkle::LeafSource::Rows {
                leaf_len: columns as u32,
            },
            log_n,
            cap_height as u32,
            None,
        );
        // Separate device hashing from the two readback measurements.
        let gpu = tree.gpu_timings(&ctx).unwrap();
        let full = || {
            let start = Instant::now();
            let result = tree.read::<F, H>(&ctx);
            (result, start.elapsed().as_secs_f64() * 1000.0)
        };
        let sparse = || {
            let start = Instant::now();
            let cap = tree.read_cap::<F, H>(&ctx, None);
            let paths = tree.read_query_paths::<F, H>(&ctx, n, cap_height, &indices);
            ((cap, paths), start.elapsed().as_secs_f64() * 1000.0)
        };
        let ((full_digests, full_cap), full_ms, (sparse_cap, paths), sparse_ms) =
            if iteration % 4 == 0 || iteration % 4 == 3 {
                let (f, fm) = full();
                let (s, sm) = sparse();
                (f, fm, s, sm)
            } else {
                let (s, sm) = sparse();
                let (f, fm) = full();
                (f, fm, s, sm)
            };
        assert_eq!(full_cap, sparse_cap);
        for (&i, path) in indices.iter().zip(paths) {
            assert_eq!(
                path,
                crate::hash::merkle_tree::merkle_tree_prove::<F, H>(
                    i,
                    n,
                    cap_height,
                    &full_digests
                )
            );
        }
        println!(
            "MERKLE_QUERY_JSON {}",
            serde_json::json!({"iteration":iteration,"full_ms":full_ms,"sparse_ms":sparse_ms,"gpu_leaves_ms":gpu.leaves_ms,"gpu_levels_ms":gpu.levels_ms,"full_digest_bytes":tree.num_digests*32,"sparse_digest_bytes":28*(log_n as usize-cap_height)*32,"cap_bytes":tree.cap_len*32,"verified":true})
        );
    }
}

/// Translates the hash module to MSL, or the quotient module when
/// `PLONKY2_GPU_MSL_MODULE=quotient`, and writes it to
/// `PLONKY2_GPU_MSL_OUT` (default `plonky2-hash.metal` in the temp dir), with
/// options matching wgpu-hal's translation with runtime checks off.
#[test]
#[ignore]
fn dump_hash_msl() {
    let source = match std::env::var("PLONKY2_GPU_MSL_MODULE").as_deref() {
        Ok("quotient") => quotient::kernel_source(Flavor::Native, 64, 2, &quotient::every_case()),
        _ => shader::hash_module(Flavor::Native),
    };
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SHADER_INT64,
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    let unchecked = naga::proc::BoundsCheckPolicy::Unchecked;
    let options = naga::back::msl::Options {
        lang_version: (3, 2),
        fake_missing_bindings: true,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: unchecked,
            buffer: unchecked,
            image_load: unchecked,
            binding_array: unchecked,
        },
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: false,
        emit_int_div_checks: false,
        ..Default::default()
    };
    let (msl, _) = naga::back::msl::write_string(
        &module,
        &info,
        &options,
        &naga::back::msl::PipelineOptions::default(),
    )
    .expect("MSL translation");
    let path = std::env::var("PLONKY2_GPU_MSL_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("plonky2-hash.metal"));
    std::fs::write(&path, &msl).expect("write MSL");
    eprintln!("wrote {} bytes of MSL to {}", msl.len(), path.display());
}

/// Timings of a wires commitment with 2^19 leaves of 139 elements.
/// Run with `cargo test --release --features gpu-profiling bench_commitment -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_commitment() {
    use std::time::Instant;

    use crate::plonk::config::PoseidonGoldilocksConfig;
    type C = PoseidonGoldilocksConfig;
    let Some(ctx) = ctx() else { return };
    let ctx = &ctx;
    let (log_d, rate_bits, columns) = (16usize, 3usize, 135usize);
    let n = 1usize << (log_d + rate_bits);
    let mut rng = StdRng::seed_from_u64(1);
    let values = random_columns(&mut rng, columns, 1 << log_d)
        .into_iter()
        .map(PolynomialValues::new)
        .collect::<Vec<_>>();
    for round in 0..3 {
        let mut timing = TimingTree::new("gpu from_values", log::Level::Info);
        let start = Instant::now();
        let batch =
            lde::from_values::<F, C, 2>(ctx, values.clone(), rate_bits, true, 4, &mut timing);
        eprintln!("round {round}: gpu from_values {:?}", start.elapsed());
        timing.print();
        drop(batch);
    }
    let leaves: Vec<Vec<F>> = (0..n)
        .map(|_| random_fields(&mut rng, columns + 4))
        .collect();
    let start = Instant::now();
    let mut words = Vec::with_capacity(n * (columns + 4));
    for leaf in &leaves {
        words.extend(leaf.iter().map(|x| x.to_noncanonical_u64()));
    }
    eprintln!("flatten {:?}", start.elapsed());
    let start = Instant::now();
    let src = ctx.upload("leaves", &words);
    eprintln!("upload {:?}", start.elapsed());
    for round in 0..3 {
        let start = Instant::now();
        let tree = merkle::build_tree(
            ctx,
            crate::plonk::config::GpuHash::Poseidon,
            &src,
            merkle::LeafSource::Rows {
                leaf_len: (columns + 4) as u32,
            },
            19,
            4,
            None,
        );
        eprintln!("round {round}: build_tree {:?}", start.elapsed());
        let start = Instant::now();
        let (digests, cap) = tree.read::<F, H>(ctx);
        eprintln!(
            "round {round}: read {:?} ({} digests, {} cap)",
            start.elapsed(),
            digests.len(),
            cap.len()
        );
    }
    let start = Instant::now();
    let tree = MerkleTree::<F, H>::new_cpu(MerkleLeaves::from_rows(leaves), 4);
    eprintln!(
        "cpu MerkleTree::new {:?} ({} digests)",
        start.elapsed(),
        tree.digests.len()
    );
}

/// The LDE of a batch (139 columns of 2^16 coefficients to 2^19
/// points) as a whole, then each stage group alone, then a zero-stage pass
/// through the same kernel (tile load and store only) as the memory-traffic
/// reference. Best of six submissions each.
/// Run with `cargo test --release --features gpu-profiling bench_fft_passes -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_fft_passes() {
    use std::time::Instant;

    use crate::gpu::lde::{fft_groups, FftJob};
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let (log_d, rate_bits, columns) = (16u32, 3u32, 139usize);
    let log_n = log_d + rate_bits;
    let n = 1usize << log_n;
    let degree = 1usize << log_d;
    let mut rng = StdRng::seed_from_u64(3);
    let coeffs = random_columns(&mut rng, columns, degree);
    let refs = coeffs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let input = lde::upload_columns(ctx, "coeffs", &refs);
    let data = ctx.storage("lde", (columns * n * 8) as u64);
    let job = |shift: bool| FftJob {
        log_n,
        columns: columns as u32,
        input: &input,
        in_len: degree as u32,
        in_stride: degree as u32,
        data: &data,
        output: None,
        shift_in: shift.then(GoldilocksField::coset_shift),
        inverse: false,
        shift_out: None,
    };
    let plan = shader::fft_plan(log_n);
    assert_eq!(plan.iter().map(|g| g.1).sum::<u32>(), log_n);
    eprintln!("plan {plan:?} (s0, stages, low_bits); {columns} columns, n = 2^{log_n}");
    let time = |label: &str, record: &dyn Fn(&mut wgpu::CommandEncoder)| {
        let mut best = f64::MAX;
        for _ in 0..6 {
            let mut encoder = ctx.encoder(label);
            record(&mut encoder);
            let start = Instant::now();
            ctx.submit(encoder);
            best = best.min(start.elapsed().as_secs_f64());
        }
        eprintln!("{label:40} best {:8.2} ms", best * 1e3);
        best
    };
    let variants = [("one tile per workgroup", 0u32), ("pipelined, 8 tiles/workgroup", 8)];
    for (label, tpw) in variants {
        time(&format!("whole LDE (coset shift in), {label}"), &|e| {
            lde::fft_with_plan(ctx, e, job(true), plan.clone(), None, tpw)
        });
        for g in 0..plan.len() {
            time(&format!("group {g} {:?}, {label}", plan[g]), &|e| {
                lde::fft_with_plan(ctx, e, job(true), plan.clone(), Some(g), tpw)
            });
        }
    }
    time("whole LDE (no shift), pipelined", &|e| {
        lde::fft_with_plan(ctx, e, job(false), plan.clone(), None, 8)
    });
    for tpw in [4u32, 16] {
        time(&format!("whole LDE, pipelined, {tpw} tiles/workgroup"), &|e| {
            lde::fft_with_plan(ctx, e, job(true), plan.clone(), None, tpw)
        });
    }
    // The IFFT of the values (2^16 per column).
    let values = random_columns(&mut rng, columns, degree);
    let refs = values.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let vals = lde::upload_columns(ctx, "values", &refs);
    let coeffs_out = ctx.storage("ifft coeffs", (columns * degree * 8) as u64);
    let work = ctx.storage("ifft work", (columns * degree * 8) as u64);
    let ifft = || FftJob {
        log_n: log_d,
        columns: columns as u32,
        input: &vals,
        in_len: degree as u32,
        in_stride: degree as u32,
        data: &work,
        output: Some(&coeffs_out),
        shift_in: None,
        inverse: true,
        shift_out: None,
    };
    let ifft_plan = shader::fft_plan(log_d);
    eprintln!("IFFT plan {ifft_plan:?}");
    for (label, tpw) in variants {
        time(&format!("IFFT whole, {label}"), &|e| {
            lde::fft_with_plan(ctx, e, ifft(), ifft_plan.clone(), None, tpw)
        });
        for g in 0..ifft_plan.len() {
            time(&format!("IFFT group {g} {:?}, {label}", ifft_plan[g]), &|e| {
                lde::fft_with_plan(ctx, e, ifft(), ifft_plan.clone(), Some(g), tpw)
            });
        }
    }
    for tpw in [4u32, 16] {
        time(&format!("IFFT whole, pipelined, {tpw} tiles/workgroup"), &|e| {
            lde::fft_with_plan(ctx, e, ifft(), ifft_plan.clone(), None, tpw)
        });
    }
    // Zero stages: every workgroup loads a contiguous tile and stores it back.
    let bl = shader::FFT_TILE_BITS;
    let params = ctx.uniform(
        "fft copy params",
        &[
            log_n,
            bl,
            0,
            bl,
            columns as u32,
            degree as u32,
            degree as u32,
            0,
            0,
            0,
            0,
            0,
        ],
    );
    let twiddles = ctx.twiddles(log_n);
    let bindings = ctx.bind_fft(&params, &input, &data, &ctx.dummy_rw, &twiddles, &ctx.dummy);
    let bytes = 2.0 * (columns * n * 8) as f64;
    let copy = time("zero-stage pass (read + write data)", &|e| {
        let mut pass = e.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.fft);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((n >> bl) as u32, columns as u32, 1);
    });
    eprintln!(
        "zero-stage pass moves {:.0} MB: {:.1} GB/s",
        bytes / 1e6,
        bytes / copy / 1e9
    );
    // The same two kernels into a buffer acquired before each submission.
    // With storage pooling disabled, the difference includes first-use costs
    // (zero initialization and page faults); pooling can reuse the allocation.
    for (label, zero_stage) in [
        ("whole LDE, fresh buffer", false),
        ("zero-stage pass, fresh buffer", true),
    ] {
        let mut times = Vec::new();
        for _ in 0..6 {
            let fresh = ctx.storage("lde fresh", (columns * n * 8) as u64);
            let mut encoder = ctx.encoder(label);
            if zero_stage {
                let bindings = ctx.bind_fft(
                    &params,
                    &input,
                    &fresh,
                    &ctx.dummy_rw,
                    &twiddles,
                    &ctx.dummy,
                );
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&ctx.fft);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups((n >> bl) as u32, columns as u32, 1);
            } else {
                let job = FftJob {
                    data: &fresh,
                    ..job(true)
                };
                fft_groups(ctx, &mut encoder, job, None);
            }
            let start = Instant::now();
            ctx.submit(encoder);
            times.push(start.elapsed().as_secs_f64() * 1e3);
            drop(fresh);
        }
        eprintln!(
            "{label:40} {:?} ms",
            times
                .iter()
                .map(|t| (t * 10.0).round() / 10.0)
                .collect::<Vec<_>>()
        );
    }
}

/// Integer throughput of the adapter and the in-register rate of the Poseidon
/// permutation, as a ceiling for the hashing kernels. Each thread runs `iters`
/// steps of eight independent operation chains (or `iters` permutations of
/// one state) with no memory traffic inside the loop.
/// Run with `cargo test --release --features gpu-profiling bench_int_throughput -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_int_throughput() {
    use std::fmt::Write;
    use std::time::Instant;

    let Some(ctx) = ctx() else { return };
    let ctx = &ctx;
    let mut source = shader::hash_module(ctx.flavor);
    // Rotating dependencies: step n+1 needs every chain's value from step n,
    // so the loop cannot be folded; the eight operations of a step are independent.
    let chains = 8;
    // Steps per loop iteration, so that the loop's own operations are amortized.
    let unroll = 16;
    let mut kernel = |name: &str, step: &dyn Fn(usize, usize) -> String| {
        writeln!(source, "\n@compute @workgroup_size(64)\nfn {name}(@builtin(global_invocation_id) id: vec3<u32>) {{").unwrap();
        source.push_str("    let t = id.x;\n    if (t >= hp.count) { return; }\n");
        for k in 0..chains {
            writeln!(source, "    var a{k} = src[t * 8u + {k}u].x;").unwrap();
        }
        source.push_str("    for (var i = 0u; i < hp.leaf_len; i++) {\n");
        for _ in 0..unroll {
            for k in 0..chains {
                writeln!(source, "        {}", step(k, (k + 1) % chains)).unwrap();
            }
        }
        source.push_str("    }\n    digests[t] = Field(");
        source.push_str(
            &(0..chains)
                .map(|k| format!("a{k}"))
                .collect::<Vec<_>>()
                .join(" ^ "),
        );
        source.push_str(", 0u);\n}\n");
    };
    kernel("bench_add", &|k, j| format!("a{k} = a{k} + a{j};"));
    kernel("bench_mul", &|k, j| format!("a{k} = a{k} * a{j};"));
    kernel("bench_cmp", &|k, j| {
        format!("a{k} = a{k} + u32(a{k} < a{j});")
    });
    kernel("bench_sel", &|k, j| {
        format!("a{k} = select(a{k}, a{j}, a{k} < a{j});")
    });
    kernel("bench_mulwide", &|k, j| {
        format!("{{ let p = mulwide32(a{k}, a{j}); a{k} = p.x ^ p.y; }}")
    });
    kernel("bench_mul_field", &|k, j| {
        format!(
            "{{ let p = mul_field(Field(a{k}, a{j}), Field(a{j}, a{k} ^ 0x5555u)); a{k} = p.x ^ p.y; }}"
        )
    });
    // Multiply-add forms: the same time as the multiply alone means the
    // compiler fuses the add into the multiplier's instruction.
    kernel("bench_mad", &|k, j| format!("a{k} = a{k} * a{j} + a{j};"));
    kernel("bench_madhi", &|k, j| {
        format!("a{k} = u32((u64(a{k}) * u64(a{j})) >> 32u) + a{j};")
    });
    // 32x32 + 32 -> 64 and 32x32 + 32 + 32 -> 64 (both fit): the same time as
    // mulwide means the multiplier absorbs the addends and their carries.
    kernel("bench_mad64", &|k, j| {
        format!("{{ let p = u64(a{k}) * u64(a{j}) + u64(a{j} ^ 0x1234u); a{k} = u32(p) ^ u32(p >> 32u); }}")
    });
    kernel("bench_mad64_2", &|k, j| {
        format!("{{ let p = u64(a{k}) * u64(a{j}) + u64(a{j} ^ 0x1234u) + u64(a{k} ^ 0x4321u); a{k} = u32(p) ^ u32(p >> 32u); }}")
    });
    kernel("bench_wide", &|k, j| {
        format!(
            "{{ let w = wide(Field(a{k}, a{j}), Field(a{j}, a{k} ^ 0x5555u)); a{k} = w.x ^ w.y ^ w.z ^ w.w; }}"
        )
    });
    kernel("bench_reduce128", &|k, j| {
        format!("{{ let r = reduce128(vec4<u32>(a{k}, a{j}, a{j} ^ a{k}, a{k} + a{j})); a{k} = r.x ^ r.y; }}")
    });
    kernel("bench_mulhi", &|k, j| {
        format!("a{k} = u32((u64(a{k}) * u64(a{j})) >> 32u);")
    });
    kernel("bench_add_wrap", &|k, j| {
        format!("{{ let p = add_wrap(Field(a{k}, a{j}), Field(a{j}, a{k})); a{k} = p.x ^ p.y; }}")
    });
    // Float chains beside integer chains. A mixed kernel whose rate is the sum
    // of the two kinds alone shows the GPU issuing them concurrently; a rate
    // between them shows a shared pipeline. The float values stay in
    // (0.11, 0.89): x * y + 0.1 maps that interval into itself.
    let mut kernel_mixed = |name: &str, step: &dyn Fn(usize) -> String| {
        writeln!(source, "\n@compute @workgroup_size(64)\nfn {name}(@builtin(global_invocation_id) id: vec3<u32>) {{").unwrap();
        source.push_str("    let t = id.x;\n    if (t >= hp.count) { return; }\n");
        for k in 0..chains {
            writeln!(source, "    var a{k} = src[t * 8u + {k}u].x;").unwrap();
            writeln!(
                source,
                "    var f{k} = 0.2 + f32(src[t * 8u + {k}u].y & 0xffffu) * (0.6 / 65536.0);"
            )
            .unwrap();
        }
        source.push_str("    for (var i = 0u; i < hp.leaf_len; i++) {\n");
        for _ in 0..unroll {
            for k in 0..chains {
                writeln!(source, "        {}", step(k)).unwrap();
            }
        }
        source.push_str("    }\n    digests[t] = Field(");
        source.push_str(
            &(0..chains)
                .map(|k| format!("a{k}"))
                .collect::<Vec<_>>()
                .join(" ^ "),
        );
        source.push_str(", bitcast<u32>(");
        source.push_str(
            &(0..chains)
                .map(|k| format!("f{k}"))
                .collect::<Vec<_>>()
                .join(" + "),
        );
        source.push_str("));\n}\n");
    };
    let fma = |k: usize| format!("f{k} = fma(f{k}, f{}, 0.1);", 4 + (k + 1) % 4);
    let iadd = |k: usize| format!("a{k} = a{k} + a{};", (k + 1) % 4);
    let imul = |k: usize| format!("a{k} = a{k} * a{};", (k + 1) % 4);
    kernel_mixed("bench_fma", &|k| {
        format!("f{k} = fma(f{k}, f{}, 0.1);", (k + 1) % 8)
    });
    kernel_mixed("bench_mixed_add_fma", &|k| if k < 4 { iadd(k) } else { fma(k) });
    kernel_mixed("bench_mixed_mul_fma", &|k| if k < 4 { imul(k) } else { fma(k) });
    kernel_mixed("bench_mixed_mul_add", &|k| {
        if k < 4 {
            imul(k)
        } else {
            format!("a{k} = a{k} + a{};", 4 + (k + 1) % 4)
        }
    });
    if ctx.flavor == Flavor::Native {
        // The same chains over 64-bit words.
        let mut kernel64 = |name: &str, step: &dyn Fn(usize, usize) -> String| {
            writeln!(source, "\n@compute @workgroup_size(64)\nfn {name}(@builtin(global_invocation_id) id: vec3<u32>) {{").unwrap();
            source.push_str("    let t = id.x;\n    if (t >= hp.count) { return; }\n");
            for k in 0..chains {
                writeln!(source, "    var a{k} = u64(src[t * 8u + {k}u].x) | (u64(src[t * 8u + {k}u].y) << 32u);").unwrap();
            }
            source.push_str("    for (var i = 0u; i < hp.leaf_len; i++) {\n");
            for _ in 0..unroll {
                for k in 0..chains {
                    writeln!(source, "        {}", step(k, (k + 1) % chains)).unwrap();
                }
            }
            source.push_str("    }\n    let r = ");
            source.push_str(
                &(0..chains)
                    .map(|k| format!("a{k}"))
                    .collect::<Vec<_>>()
                    .join(" ^ "),
            );
            source.push_str(";\n    digests[t] = Field(u32(r), u32(r >> 32u));\n}\n");
        };
        kernel64("bench_add64", &|k, j| format!("a{k} = a{k} + a{j};"));
        kernel64("bench_cmp64", &|k, j| {
            format!("a{k} = select(a{k}, a{j}, a{k} < a{j});")
        });
        kernel64("bench_mul64", &|k, j| format!("a{k} = a{k} * a{j};"));
        kernel64("bench_mulwide64", &|k, j| {
            format!("a{k} = (a{k} & u64(0xffffffffu)) * (a{j} >> 32u);")
        });
    }
    source.push_str(
        "\n@compute @workgroup_size(64)\nfn bench_permute(@builtin(global_invocation_id) id: vec3<u32>) {\n\
         \x20   let t = id.x;\n    if (t >= hp.count) { return; }\n    let b = 12u * t;\n\
         \x20   var s = State(src[b], src[b + 1u], src[b + 2u], src[b + 3u], src[b + 4u], src[b + 5u],\n\
         \x20       src[b + 6u], src[b + 7u], src[b + 8u], src[b + 9u], src[b + 10u], src[b + 11u]);\n\
         \x20   for (var i = 0u; i < hp.leaf_len; i++) { s = permute(s); }\n\
         \x20   digests[t] = s.s0;\n}\n",
    );
    let module = module(&ctx.device, "bench", &source);
    let threads = 1u32 << 20;
    let mut rng = StdRng::seed_from_u64(7);
    let words = (0..12 * threads as usize)
        .map(|_| rng.next_u64() % crate::field::goldilocks_field::GoldilocksField::ORDER)
        .collect::<Vec<_>>();
    let src = ctx.upload("bench src", &words);
    let out = ctx.storage("bench out", threads as u64 * 8);
    // (entry, iterations, operations per chain step, chains)
    let kernels = [
        ("bench_add", 256u32, 1.0, chains * unroll),
        ("bench_mul", 256, 1.0, chains * unroll),
        ("bench_cmp", 256, 2.0, chains * unroll),
        ("bench_sel", 256, 2.0, chains * unroll),
        ("bench_mulwide", 256, 1.0, chains * unroll),
        ("bench_mul_field", 64, 1.0, chains * unroll),
        ("bench_mad", 256, 1.0, chains * unroll),
        ("bench_madhi", 256, 1.0, chains * unroll),
        ("bench_mad64", 256, 1.0, chains * unroll),
        ("bench_mad64_2", 256, 1.0, chains * unroll),
        ("bench_wide", 128, 1.0, chains * unroll),
        ("bench_reduce128", 256, 1.0, chains * unroll),
        ("bench_mulhi", 256, 1.0, chains * unroll),
        ("bench_add_wrap", 256, 1.0, chains * unroll),
        ("bench_fma", 256, 1.0, chains * unroll),
        ("bench_mixed_add_fma", 256, 1.0, chains * unroll),
        ("bench_mixed_mul_fma", 256, 1.0, chains * unroll),
        ("bench_mixed_mul_add", 256, 1.0, chains * unroll),
        ("bench_add64", 256, 1.0, chains * unroll),
        ("bench_cmp64", 256, 1.0, chains * unroll),
        ("bench_mul64", 256, 1.0, chains * unroll),
        ("bench_mulwide64", 256, 1.0, chains * unroll),
        ("bench_permute", 16, 1.0, 1),
    ];
    // PLONKY2_GPU_BENCH_ONLY=name,name selects entries.
    let only = std::env::var("PLONKY2_GPU_BENCH_ONLY").unwrap_or_default();
    for (entry, iters, ops_per_step, chains) in kernels {
        if entry.ends_with("64") && ctx.flavor != Flavor::Native {
            continue;
        }
        if !only.is_empty() && !only.split(',').any(|o| o == entry) {
            continue;
        }
        let pipeline = pipeline(&ctx.device, &ctx.hash_layout, &module, entry);
        let params = ctx.uniform(
            "bench params",
            &[threads, iters, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let bindings = ctx.bind(
            &ctx.hash_layout,
            &[&params, &src, &out, &ctx.dummy_rw, &ctx.dummy_rw],
        );
        let mut best = f64::MAX;
        for _ in 0..6 {
            let mut encoder = ctx.encoder(entry);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(threads / 64, 1, 1);
            }
            let start = Instant::now();
            ctx.submit(encoder);
            best = best.min(start.elapsed().as_secs_f64());
        }
        let steps = threads as f64 * iters as f64 * chains as f64;
        eprintln!(
            "{entry:16} best {:8.3} ms  {:8.1} M steps/ms  {:7.3} T step-ops/s",
            best * 1e3,
            steps / best / 1e9,
            steps * ops_per_step / best / 1e12
        );
    }
}

/// Op histogram of a dumped quotient program and the interpreter's cost on it
/// (`PLONKY2_GPU_PROGRAM`
/// names the file written by `PLONKY2_GPU_PROGRAM_DUMP`; see
/// `quotient::dump_program`). Random inputs of the circuit's shape; timing
/// does not depend on the values.
/// Run with `PLONKY2_GPU_PROGRAM=... cargo test --release --features gpu-profiling bench_quotient_program -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_quotient_program() {
    use std::time::Instant;

    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let path = std::env::var("PLONKY2_GPU_PROGRAM").expect("PLONKY2_GPU_PROGRAM");
    let text = std::fs::read_to_string(&path).expect("program dump");
    let mut lines = text.lines().map(|line| {
        line.split_whitespace()
            .map(|w| w.parse::<u32>().unwrap())
            .collect::<Vec<_>>()
    });
    let params = lines.next().unwrap();
    let columns = lines.next().unwrap();
    // The executable ops end at the end marker; the round constant blocks
    // of the packed MDS ops follow it in the same buffer.
    let all_ops = lines
        .map(|op| [op[0], op[1], op[2], op[3]])
        .collect::<Vec<_>>();
    let num_ops = all_ops
        .iter()
        .position(|op| op[0] & quotient::BASE_MASK == 21)
        .unwrap_or(all_ops.len());
    let ops = &all_ops[..num_ops];
    let (n, n_lde) = (params[0] as usize, params[2] as usize);
    let (cs_cols, w_cols, z_cols, chal_len, nc, registers) = (
        columns[0] as usize,
        columns[1] as usize,
        columns[2] as usize,
        columns[3] as usize,
        columns[4],
        columns[5],
    );
    assert_eq!(params[9] as usize, ops.len());
    eprintln!(
        "{} ops, {registers} registers, {nc} challenges, n={n}, n_lde={n_lde}, columns cs={cs_cols} w={w_cols} z={z_cols}",
        ops.len()
    );

    // Histogram by opcode.
    let names = [
        "wire",
        "constant",
        "public_input",
        "literal",
        "add",
        "sub",
        "mul",
        "filter",
        "weighted",
        "mulc",
        "addc",
        "muladd",
        "finish",
        "mds",
        "mov",
        "square",
        "sbox",
        "qmul",
        "mulsub",
        "mulrsub",
        "mdsp",
        "end",
        "addw",
        "subw",
        "mulw",
        "muladdw",
        "mulsubw",
        "mulrsubw",
    ];
    let mut counts = vec![0usize; names.len()];
    for op in ops {
        counts[(op[0] & quotient::BASE_MASK) as usize] += 1;
    }
    for (name, count) in names.iter().zip(&counts) {
        eprintln!("  {name:13} {count:6}");
    }

    // Buffers of the circuit's shape with random canonical values.
    let mut rng = StdRng::seed_from_u64(0x51504f47);
    let mut random = |len: usize| {
        (0..len)
            .map(|_| rng.next_u64() % GoldilocksField::ORDER)
            .collect::<Vec<u64>>()
    };
    let cs_lde = ctx.upload("cs", &random(cs_cols * n_lde));
    let w_lde = ctx.upload("w", &random(w_cols * n_lde));
    let z_lde = ctx.upload("z", &random(z_cols * n_lde));
    let row_table = ctx.upload("rows", &random(3 * n));
    let chal = ctx.upload("chal", &random(chal_len));
    let out = ctx.storage("out", (nc as usize * n * 8) as u64);
    let cases = quotient::case_list(&all_ops);
    eprintln!("{} opcodes in frequency order: {:?}", cases.len(), cases);
    let upload_ops = |ops: &[[u32; 4]], cases: &[u32]| -> wgpu::Buffer {
        let bytes = quotient::program_words(ops, cases)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ops"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
    };
    let ops_buffer = upload_ops(&all_ops, &cases);
    let layout = storage_layout(
        &ctx.device,
        "quotient bench",
        &[
            (true, true),
            (false, true),
            (false, true),
            (false, true),
            (false, true),
            (false, true),
            (false, true),
            (false, false),
        ],
    );
    let time = |pipeline: &wgpu::ComputePipeline, ops_buffer: &wgpu::Buffer, num_ops: u32| -> f64 {
        let rows = n as u32;
        let mut params = params.clone();
        params[1] = rows;
        params[9] = num_ops;
        let uniform = ctx.uniform("params", &params);
        let bindings = ctx.bind(
            &layout,
            &[
                &uniform, &cs_lde, &w_lde, &z_lde, &row_table, &chal, ops_buffer, &out,
            ],
        );
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut encoder = ctx.encoder("quotient bench");
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(rows.div_ceil(shader::HASH_WORKGROUP), 1, 1);
            }
            let start = Instant::now();
            ctx.submit(encoder);
            best = best.min(start.elapsed().as_secs_f64());
        }
        best
    };
    let compile = |label: &str, source: &str| -> wgpu::ComputePipeline {
        let start = Instant::now();
        let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = module(&ctx.device, label, source);
        let pipeline = pipeline(&ctx.device, &layout, &module, "quotient");
        if let Some(error) = pollster::block_on(validation.pop()) {
            panic!("{label}: {error}");
        }
        // wgpu compiles Metal pipelines lazily: the first dispatch pays.
        let created = start.elapsed();
        let start = Instant::now();
        time(&pipeline, &ops_buffer, 0);
        eprintln!(
            "{label}: {} KB, pipeline {created:?}, first dispatch {:?}",
            source.len() / 1024,
            start.elapsed()
        );
        pipeline
    };

    let interpreter = compile(
        "interpreter",
        &quotient::kernel_source(ctx.flavor, registers, nc, &cases),
    );
    let prologue = time(&interpreter, &ops_buffer, 0);
    eprintln!("interpreter: prologue (0 ops) {:.3} ms", prologue * 1e3);
    let prefixes = [ops.len()];
    for &num_ops in &prefixes {
        let t = time(&interpreter, &ops_buffer, num_ops as u32);
        eprintln!(
            "interpreter: {num_ops:6} ops {:8.3} ms, {:.1} ps per point per op beyond the prologue",
            t * 1e3,
            (t - prologue) / n as f64 / num_ops as f64 * 1e12
        );
    }
    // PLONKY2_GPU_PROGRAM_VARIANTS=path,path: the same program in other
    // orders or register assignments (the register count may differ), timed
    // in alternation with the reference, each output compared with its.
    let variants = std::env::var("PLONKY2_GPU_PROGRAM_VARIANTS")
        .map(|v| v.split(',').map(str::to_owned).collect::<Vec<_>>())
        .unwrap_or_default();
    if !variants.is_empty() {
        let out_bytes = nc as u64 * n as u64 * 8;
        let output = |pipeline: &wgpu::ComputePipeline, buffer: &wgpu::Buffer, num_ops: u32| {
            time(pipeline, buffer, num_ops);
            ctx.read(&out, out_bytes, |b| b.to_vec())
        };
        let reference = output(&interpreter, &ops_buffer, ops.len() as u32);
        let mut owned: Vec<(wgpu::ComputePipeline, wgpu::Buffer)> = Vec::new();
        let mut labels = vec!["reference".to_owned()];
        let mut nums = vec![ops.len()];
        for path in &variants {
            let text = std::fs::read_to_string(path).expect("variant dump");
            let mut lines = text.lines().map(|line| {
                line.split_whitespace()
                    .map(|w| w.parse::<u32>().unwrap())
                    .collect::<Vec<_>>()
            });
            let vparams = lines.next().unwrap();
            let vcolumns = lines.next().unwrap();
            assert_eq!(vparams, params, "{path}: parameters differ");
            let vops = lines
                .map(|op| [op[0], op[1], op[2], op[3]])
                .collect::<Vec<_>>();
            let vnum = vops
                .iter()
                .position(|op| op[0] & quotient::BASE_MASK == 21)
                .unwrap_or(vops.len());
            let vregs = vcolumns[5];
            let label = format!("{} ({} ops, {vregs} registers)", path.rsplit('/').next().unwrap(), vnum);
            let vcases = quotient::case_list(&vops);
            let pipeline = compile(
                &label,
                &quotient::kernel_source(ctx.flavor, vregs, nc, &vcases),
            );
            let buffer = upload_ops(&vops, &vcases);
            let got = output(&pipeline, &buffer, vnum as u32);
            let same = got == reference;
            eprintln!("{label}: output {}", if same { "matches" } else { "DIFFERS" });
            owned.push((pipeline, buffer));
            labels.push(label);
            nums.push(vnum);
        }
        let entries = std::iter::once((&interpreter, &ops_buffer))
            .chain(owned.iter().map(|(p, b)| (p, b)))
            .collect::<Vec<_>>();
        let mut best = vec![f64::MAX; entries.len()];
        for _ in 0..4 {
            for (k, (pipeline, buffer)) in entries.iter().enumerate() {
                best[k] = best[k].min(time(pipeline, buffer, nums[k] as u32));
            }
        }
        for (k, label) in labels.iter().enumerate() {
            eprintln!(
                "{label}: {:8.3} ms ({:+.1}% vs reference)",
                best[k] * 1e3,
                (best[k] / best[0] - 1.0) * 100.0
            );
        }
    }

    // Gate boundaries: the ops up to each `finish`.
    let mut gates = Vec::new();
    let mut gate_start = 0;
    for (i, op) in ops.iter().enumerate() {
        if op[0] & quotient::BASE_MASK == 12 {
            gates.push(gate_start..i + 1);
            gate_start = i + 1;
        }
    }
    // The interpreter on the program split into consecutive chunks at gate
    // boundaries: a sum below the single dispatch means the long program
    // itself costs (program cache, register pressure), not the gates.
    for parts in [2usize, 4, 8] {
        let per_part = ops.len().div_ceil(parts);
        let mut chunks: Vec<std::ops::Range<usize>> = Vec::new();
        let mut chunk_start = 0;
        for range in &gates {
            if range.end - chunk_start >= per_part || range.end == ops.len() {
                chunks.push(chunk_start..range.end);
                chunk_start = range.end;
            }
        }
        let mut raw = 0.0;
        for chunk in &chunks {
            let buffer = upload_ops(&ops[chunk.clone()], &cases);
            raw += time(&interpreter, &buffer, chunk.len() as u32);
        }
        eprintln!(
            "interpreter in {} chunks ({:?} ops): {:.3} ms raw, {:.3} ms beyond {} prologues; whole program {:.3} ms beyond one",
            chunks.len(),
            chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            raw * 1e3,
            (raw - chunks.len() as f64 * prologue) * 1e3,
            chunks.len(),
            (time(&interpreter, &ops_buffer, ops.len() as u32) - prologue) * 1e3
        );
    }
}

#[test]
#[ignore = "timing probe; run alone"]
fn bench_opening_fft_crossover() {
    use std::time::Instant;

    use crate::field::extension::quadratic::QuadraticExtension;
    let ctx = context().expect("test requires a GPU");
    for log_n in 10..=19 {
        let coeffs = PolynomialCoeffs::new(
            (0..1 << log_n)
                .map(|i| QuadraticExtension([F::from_canonical_usize(i), F::ONE]))
                .collect(),
        );
        let mut cpu = Vec::new();
        let mut gpu = Vec::new();
        for repeat in 0..8 {
            let start = Instant::now();
            let expected = coeffs.coset_fft(F::coset_shift().into());
            let cpu_time = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let actual = fri::try_opening_fft_from::<F, 2>(&coeffs, 1).unwrap();
            let gpu_time = start.elapsed().as_secs_f64();
            assert_eq!(actual.values, expected.values);
            if repeat > 0 {
                cpu.push(cpu_time);
                gpu.push(gpu_time);
            }
        }
        cpu.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        println!(
            "opening_fft log_n={log_n} cpu={:.6} gpu={:.6}",
            cpu[3], gpu[3]
        );
    }
    drop(ctx);
    shutdown().unwrap();
}

/// Transfer probes for discrete GPUs, at the
/// size of 135 columns of 2^16 words (about 70 MB) and smaller:
/// device to readback-buffer copies (what a commitment's coefficient readback
/// costs on the queue), device to device copies for reference, and three
/// ways to fill a device buffer from the host (a fresh buffer mapped at
/// creation; a reused mapped staging buffer, as the witness gather uses, and a
/// copy; a queue write). Wall times of submit-and-wait, best of five.
/// Run alone: `cargo test --release --features gpu-profiling
/// bench_transfer_paths -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_transfer_paths() {
    use std::time::Instant;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    eprintln!(
        "adapter {:?}, mappable {}",
        ctx.device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
        ctx.mappable
    );
    let best = |mut f: Box<dyn FnMut() -> f64 + '_>| (0..5).map(|_| f()).fold(f64::MAX, f64::min);
    for mb in [8usize, 70, 280] {
        let bytes = (mb << 20) as u64;
        let words = vec![0x1234_5678_9abc_def0u64; mb << 17];
        let src = ctx.storage("transfer src", bytes);
        ctx.write(&src, 0, &words);
        let dst = ctx.storage("transfer dst", bytes);
        let staging = ctx.readback_buffer(bytes);
        let copy = |to: &wgpu::Buffer| -> f64 {
            let mut encoder = ctx.encoder("transfer copy");
            encoder.copy_buffer_to_buffer(&src, 0, to, 0, bytes);
            let start = Instant::now();
            ctx.submit(encoder);
            start.elapsed().as_secs_f64() * 1e3
        };
        let to_readback = best(Box::new(|| copy(&staging)));
        let to_device = best(Box::new(|| copy(&dst)));
        let decode = best(Box::new(|| {
            // The staged read maps the readback buffer itself, as commitments do.
            let mut encoder = ctx.encoder("transfer copy");
            encoder.copy_buffer_to_buffer(&src, 0, &staging, 0, bytes);
            let done = ctx.submit_async(encoder);
            ctx.wait(&done);
            let start = Instant::now();
            let sum = ctx.read_staged_with_timing(staging.clone(), bytes, &done, None, |b| {
                b.iter().map(|&x| x as u64).sum::<u64>()
            });
            std::hint::black_box(sum);
            start.elapsed().as_secs_f64() * 1e3
        }));
        // Host fills: a fresh mapped-at-creation buffer written per column in parallel.
        let columns = 135usize;
        let fresh = best(Box::new(|| {
            let start = Instant::now();
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("transfer fresh mapped"),
                size: bytes,
                usage: ctx.storage_usage(),
                mapped_at_creation: true,
            });
            let column = (bytes as usize / columns / 8) * 8;
            (0..columns).into_par_iter().for_each(|c| {
                let start = (c * column) as u64;
                let mut view = buffer
                    .slice(start..start + column as u64)
                    .get_mapped_range_mut()
                    .expect("mapped");
                let (words, _) = view.slice(..).into_chunks::<8>();
                for cell in words {
                    cell.write([0x5a; 8]);
                }
            });
            buffer.unmap();
            let mut encoder = ctx.encoder("transfer use");
            encoder.copy_buffer_to_buffer(&buffer, 0, &dst, 0, 16);
            ctx.submit(encoder);
            start.elapsed().as_secs_f64() * 1e3
        }));
        // A reused mapped staging buffer, refilled and copied each time.
        let upload = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transfer staging upload"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let reused = best(Box::new(|| {
            let (send, receive) = std::sync::mpsc::channel();
            upload.slice(..).map_async(wgpu::MapMode::Write, move |r| {
                let _ = send.send(r);
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            receive.recv().unwrap().unwrap();
            let start = Instant::now();
            let column = (bytes as usize / columns / 8) * 8;
            (0..columns).into_par_iter().for_each(|c| {
                let start = (c * column) as u64;
                let mut view = upload
                    .slice(start..start + column as u64)
                    .get_mapped_range_mut()
                    .expect("mapped");
                let (words, _) = view.slice(..).into_chunks::<8>();
                for cell in words {
                    cell.write([0x5a; 8]);
                }
            });
            upload.unmap();
            let mut encoder = ctx.encoder("transfer staging copy");
            encoder.copy_buffer_to_buffer(&upload, 0, &dst, 0, bytes);
            ctx.submit(encoder);
            start.elapsed().as_secs_f64() * 1e3
        }));
        let queued = best(Box::new(|| {
            let start = Instant::now();
            ctx.write(&dst, 0, &words);
            let encoder = ctx.encoder("transfer queue write");
            ctx.submit(encoder);
            start.elapsed().as_secs_f64() * 1e3
        }));
        let gbps = |ms: f64| bytes as f64 / ms / 1e6;
        eprintln!(
            "TRANSFER {mb} MB: device->readback {to_readback:.2} ms ({:.1} GB/s), device->device {to_device:.2} ms ({:.1} GB/s), map+read readback {decode:.2} ms, \
             fill fresh mapped-at-creation {fresh:.2} ms ({:.1} GB/s), fill reused staging + copy {reused:.2} ms ({:.1} GB/s), queue write {queued:.2} ms ({:.1} GB/s)",
            gbps(to_readback), gbps(to_device), gbps(fresh), gbps(reused), gbps(queued)
        );
    }
}

/// Poseidon BN128: the in-register permutation rate of each flavour at
/// several workgroup sizes (each thread permutes one state `ITERS` times), then
/// trees of 2^16 leaves of 135 elements, the size of a wrap proof's wires
/// commitment, and of single elements, whose leaves need no permutation.
/// Run alone: `cargo test --release --features gpu-profiling,poseidon_bn128
/// bench_poseidon_bn128 -- --ignored --nocapture`.
#[cfg(feature = "poseidon_bn128")]
#[test]
#[ignore]
fn bench_poseidon_bn128() {
    use std::time::Instant;

    use crate::gpu::poseidon_bn128 as kernels;
    use crate::hash::poseidon_bn128::PoseidonBN128Hash;
    use crate::plonk::config::GpuHash;
    const ITERS: u32 = 8;
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let threads = 1u32 << 16;
    let layout = storage_layout(
        &ctx.device,
        "bn128 bench",
        &[
            (true, true),
            (false, true),
            (false, false),
            (false, false),
            (false, false),
            (true, true),
        ],
    );
    // Word 0 of each element is random and the top word zero, so every
    // element is below p.
    let mut rng = StdRng::seed_from_u64(0x42454e4348);
    let words = (0..threads as usize * 16)
        .map(|i| if i % 4 == 3 { 0 } else { rng.next_u64() })
        .collect::<Vec<_>>();
    let src = ctx.upload("bn128 bench states", &words);
    let out = ctx.storage("bn128 bench out", threads as u64 * 4);
    let params = ctx.uniform("bn128 bench params", &[threads, ITERS, 0, 0]);
    let constants = ctx.uniform("bn128 constants", &kernels::constant_words());
    let bindings = ctx.bind(
        &layout,
        &[
            &params,
            &src,
            &out,
            &ctx.dummy_rw,
            &ctx.dummy_rw,
            &constants,
        ],
    );
    let mut variants = Vec::new();
    for flavor in super::inversion::flavors(ctx) {
        for size in [32u32, 64, 128, 256] {
            let source = kernels::permutation_source(flavor)
                + &format!(
                    r#"
struct Params {{ count: u32, iters: u32, pad1: u32, pad2: u32 }}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<u32>;

@compute @workgroup_size({size})
fn bench(@builtin(global_invocation_id) id: vec3<u32>) {{
    let t = id.x;
    if (t >= params.count) {{ return; }}
    var s: array<Fr, 4>;
    for (var k = 0u; k < 4u; k++) {{
        for (var j = 0u; j < 8u; j++) {{ s[k][j] = inputs[32u * t + 8u * k + j]; }}
    }}
    for (var i = 0u; i < params.iters; i++) {{ s = permute(s); }}
    outputs[t] = s[0][0] ^ s[1][1] ^ s[2][2] ^ s[3][3];
}}
"#
                );
            let label = format!("{flavor:?} workgroup {size}");
            let module = module(&ctx.device, &label, &source);
            variants.push((
                label,
                size,
                pipeline(&ctx.device, &layout, &module, "bench"),
            ));
        }
    }
    let time = |pipeline: &wgpu::ComputePipeline, size: u32| {
        let mut encoder = ctx.encoder("bn128 bench");
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(threads.div_ceil(size), 1, 1);
        }
        let start = Instant::now();
        ctx.submit(encoder);
        start.elapsed().as_secs_f64()
    };
    for (_, size, pipeline) in &variants {
        time(pipeline, *size);
    }
    let mut best = vec![f64::MAX; variants.len()];
    for round in 0..6 {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..variants.len()).collect()
        } else {
            (0..variants.len()).rev().collect()
        };
        for i in order {
            best[i] = best[i].min(time(&variants[i].2, variants[i].1));
        }
    }
    for ((label, _, _), best) in variants.iter().zip(&best) {
        let perms = f64::from(threads) * f64::from(ITERS);
        eprintln!(
            "BN128_PERMUTE {label:24} best {:8.2} ms  {:6.2} M permutations/s",
            best * 1e3,
            perms / best / 1e6
        );
    }

    for leaf_len in [135u32, 1] {
        let log_leaves = 16;
        let leaves = random_fields(&mut rng, (leaf_len as usize) << log_leaves);
        let src = ctx.upload(
            "bn128 bench leaves",
            &leaves
                .iter()
                .map(|x| x.to_noncanonical_u64())
                .collect::<Vec<_>>(),
        );
        let mut best = f64::MAX;
        for _ in 0..6 {
            let start = Instant::now();
            let tree = merkle::build_tree(
                ctx,
                GpuHash::PoseidonBn128,
                &src,
                merkle::LeafSource::Rows { leaf_len },
                log_leaves,
                4,
                None,
            );
            tree.read::<F, PoseidonBN128Hash>(ctx);
            best = best.min(start.elapsed().as_secs_f64());
        }
        eprintln!(
            "BN128_TREE 2^{log_leaves} leaves x {leaf_len}: best {:.2} ms",
            best * 1e3
        );
    }
}
