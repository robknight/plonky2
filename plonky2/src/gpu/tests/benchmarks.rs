//! Optional GPU profiling probes; run individually with `gpu-profiling`.

use super::*;

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

/// Writes the MSL that wgpu's Metal backend generates for the hash module to
/// `PLONKY2_GPU_MSL_OUT` (default `plonky2-hash.metal` in the temp dir), with
/// the options wgpu-hal passes for modules created with runtime checks off.
#[test]
#[ignore]
fn dump_hash_msl() {
    let source = match std::env::var("PLONKY2_GPU_MSL_MODULE").as_deref() {
        Ok("quotient") => quotient::kernel_source(Flavor::Native, 64, 2, true),
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

/// Timings of the pieces of a MainPod-sized wires commitment (2^19 leaves of 139 elements).
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

/// The LDE of the wires batch (139 columns of 2^16 coefficients to 2^19
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
    time("whole LDE (coset shift in)", &|e| {
        fft_groups(ctx, e, job(true), None)
    });
    time("whole LDE (no shift)", &|e| {
        fft_groups(ctx, e, job(false), None)
    });
    for g in 0..plan.len() {
        time(&format!("group {g} {:?}", plan[g]), &|e| {
            fft_groups(ctx, e, job(true), Some(g))
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
    // The same two kernels into a buffer created just before each submission,
    // as the prover does once per commitment: the difference is the cost of
    // first use (zero initialization and page faults).
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
        let bindings = ctx.bind_hash(&params, &src, &out, &ctx.dummy_rw, &ctx.dummy_rw);
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

/// First-dispatch (Metal compile) time of straight-line kernels by size, with
/// the driver cache defeated. Run with
/// `cargo test --release --features gpu-profiling bench_compile -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_compile() {
    use std::time::Instant;

    use crate::plonk::circuit_data::CircuitConfig;
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let layout = storage_layout(
        &ctx.device,
        "compile bench",
        &[
            (true, true),
            (false, true),
            (false, false),
            (false, false),
            (false, false),
        ],
    );
    let count = 256u32;
    let src = ctx.upload("inputs", &vec![3u64; count as usize * 256]);
    let out = ctx.storage("outputs", count as u64 * 256 * 8);
    let dummy = ctx.storage("dummy", 16);
    let dummy2 = ctx.storage("dummy2", 16);
    let params = ctx.uniform("params", &[count, 256, 135, 0]);
    let run = |label: &str, body: &str, serial: u64| {
        let source = format!(
            "{}\nstruct TParams {{ count: u32, stride: u32, num_wires: u32, pad: u32 }}\n\
             @group(0) @binding(0) var<uniform> tp: TParams;\n\
             @group(0) @binding(1) var<storage, read> src: array<Field>;\n\
             @group(0) @binding(2) var<storage, read_write> out: array<Field>;\n\
             @group(0) @binding(3) var<storage, read_write> pih: array<Field>;\n\
             @group(0) @binding(4) var<storage, read_write> unused: array<Field>;\n\
             fn wire(row: u32, i: u32) -> Field {{ return src[row * tp.stride + i]; }}\n\
             fn constant(row: u32, j: u32) -> Field {{ return src[row * tp.stride + tp.num_wires + j]; }}\n\
             fn public_input(k: u32) -> Field {{ return pih[k]; }}\n\
             {body}\n// nonce {nonce} {serial}\n",
            shader::prelude_with_mds(ctx.flavor) + shader::QMUL
        );
        eprintln!("{label}: compiling {} KB source", source.len() / 1024);
        let start = Instant::now();
        let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = module(&ctx.device, label, &source);
        let pipeline = pipeline(&ctx.device, &layout, &module, "bench");
        if let Some(error) = pollster::block_on(validation.pop()) {
            panic!("{label}: {error}");
        }
        let created = start.elapsed();
        let bindings = ctx.bind(&layout, &[&params, &src, &out, &dummy, &dummy2]);
        let start = Instant::now();
        let mut encoder = ctx.encoder(label);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        ctx.submit(encoder);
        let first = start.elapsed();
        let start = Instant::now();
        let mut encoder = ctx.encoder(label);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        ctx.submit(encoder);
        let second = start.elapsed();
        eprintln!(
            "{label}: {} KB source, pipeline {created:?}, first dispatch {first:?}, second {second:?}",
            source.len() / 1024
        );
    };
    // Synthetic chains of alternating multiplications and additions.
    for &ops in &[250usize, 500, 1000, 2000, 4000, 8000, 16000] {
        let mut body = String::from(
            "@compute @workgroup_size(64)\nfn bench(@builtin(global_invocation_id) id: vec3<u32>) {\n    let row = id.x;\n    if (row >= tp.count) { return; }\n    let v0 = wire(row, 0u);\n    let v1 = wire(row, 1u);\n",
        );
        for i in 2..ops + 2 {
            let op = if i % 2 == 0 { "mul_field" } else { "add_field" };
            body.push_str(&format!("    let v{i} = {op}(v{}, v{});\n", i - 1, i - 2));
        }
        body.push_str(&format!("    out[row] = v{};\n}}\n", ops + 1));
        run(&format!("chain {ops}"), &body, ops as u64);
    }
    // Real gates, straight-line.
    let config = CircuitConfig::standard_recursion_config();
    for gate in traced_gates() {
        let mut trace = trace::trace_gate(&gate, &config, 0).unwrap();
        trace::fuse_quintic_products(&mut trace);
        let nc = gate.0.num_constraints();
        let body = format!(
            "{}\n@compute @workgroup_size(64)\nfn bench(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let row = id.x;\n    if (row >= tp.count) {{ return; }}\n    var cs: array<Field, {nc}>;\n    gate_0(row, Field(1u, 0u), &cs);\n    for (var k = 0u; k < {nc}u; k++) {{ out[row * {nc}u + k] = cs[k]; }}\n}}\n",
            trace::emit_gate(&trace, "gate_0", nc)
        );
        let label = gate.0.id();
        let label = label.split(['{', '(']).next().unwrap().to_owned();
        run(
            &format!("{label} ({} nodes)", trace.dag.nodes.len()),
            &body,
            trace.dag.nodes.len() as u64,
        );
    }
}

/// Op histogram of a dumped quotient program and the interpreter's cost
/// against straight-line kernels of prefixes of it (`PLONKY2_GPU_PROGRAM`
/// names the file written by `PLONKY2_GPU_PROGRAM_DUMP`; see
/// `quotient::dump_program`). Random inputs of the circuit's shape; timing
/// does not depend on the values.
/// Run with `PLONKY2_GPU_PROGRAM=... cargo test --release --features gpu-profiling bench_quotient_program -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_quotient_program() {
    use std::fmt::Write;
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
    let ops = lines
        .map(|op| [op[0], op[1], op[2], op[3]])
        .collect::<Vec<_>>();
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
    ];
    let mut counts = vec![0usize; names.len()];
    for op in &ops {
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
    let upload_ops = |ops: &[[u32; 4]]| -> wgpu::Buffer {
        let bytes = quotient::program_words(ops)
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
    let ops_buffer = upload_ops(&ops);
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
    let time_rows = |pipeline: &wgpu::ComputePipeline,
                     ops_buffer: &wgpu::Buffer,
                     num_ops: u32,
                     rows: u32|
     -> f64 {
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
    let time = |pipeline: &wgpu::ComputePipeline, ops_buffer: &wgpu::Buffer, num_ops: u32| -> f64 {
        time_rows(pipeline, ops_buffer, num_ops, n as u32)
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
        &quotient::kernel_source(ctx.flavor, registers, nc, quotient::uses_qmul(&ops)),
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

    // Straight-line kernels of prefixes: registers as named variables.
    let emit = |ops: &[[u32; 4]]| -> String {
        let mut body = String::new();
        for r in 0..registers {
            writeln!(body, "    var r{r} = Field(0u, 0u);").unwrap();
        }
        writeln!(body, "    var gate_filter = Field(1u, 0u);").unwrap();
        for d in 0..nc {
            writeln!(body, "    var h{d} = Field(0u, 0u);").unwrap();
        }
        for op in ops {
            let (code, dst, y, z, w) =
                (op[0] & quotient::BASE_MASK, op[0] >> 8, op[1], op[2], op[3]);
            let wires = (op[0] & 0xff) >> 5;
            let src = |slot: u32, s: u32| -> String {
                if wires >> slot & 1 != 0 {
                    format!("wire(row, {s}u)")
                } else {
                    format!("r{s}")
                }
            };
            let (sy, sz, sw) = (src(0, y), src(1, z), src(2, w));
            let line = match code {
                0 => format!("r{dst} = wire(row, {y}u);"),
                1 => format!("r{dst} = constant(row, {y}u);"),
                2 => format!("r{dst} = chal[PIH_OFFSET + {y}u];"),
                3 => format!("r{dst} = Field({y}u, {z}u);"),
                4 => format!("r{dst} = add_field({sy}, {sz});"),
                5 => format!("r{dst} = sub_field({sy}, {sz});"),
                6 => format!("r{dst} = mul_field({sy}, {sz});"),
                7 => {
                    let mut s = format!("gate_filter = {sy};");
                    for d in 0..nc {
                        write!(s, " h{d} = Field(0u, 0u);").unwrap();
                    }
                    s
                }
                8 => (0..nc)
                    .map(|d| format!("h{d} = mul_add(h{d}, {sy}, alpha_pow({d}u, {z}u));"))
                    .collect::<Vec<_>>()
                    .join(" "),
                9 => format!("r{dst} = mul_small({sy}, {z}u);"),
                10 => format!("r{dst} = add_const({sy}, Field({z}u, {w}u));"),
                11 => format!("r{dst} = mul_add({sw}, {sy}, {sz});"),
                18 => format!("r{dst} = mul_sub({sw}, {sy}, {sz});"),
                19 => format!("r{dst} = mul_rsub({sw}, {sy}, {sz});"),
                12 => (0..nc)
                    .map(|d| format!("acc[{d}] = mul_add(acc[{d}], h{d}, gate_filter);"))
                    .collect::<Vec<_>>()
                    .join(" "),
                13 => {
                    let inputs = (0..12)
                        .map(|k| format!("r{}", y + k))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let mut s = format!("{{ let m = mds12(array<Field, 12>({inputs}));");
                    for k in 0..12 {
                        write!(s, " r{} = m[{k}];", y + k).unwrap();
                    }
                    s.push_str(" }");
                    s
                }
                14 => format!("r{dst} = {sy};"),
                15 => format!("r{dst} = square_field({sy});"),
                16 => format!("r{dst} = sbox({sy});"),
                _ => unreachable!(),
            };
            writeln!(body, "    {line}").unwrap();
        }
        body
    };
    // Per gate (the ops up to each `finish`): the interpreter on that range
    // alone against a straight-line kernel of it, both beyond their own
    // prologue (the empty kernels differ: the interpreter's register array
    // costs occupancy even when the loop runs zero times).
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
            let buffer = upload_ops(&ops[chunk.clone()]);
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
    if std::env::var("PLONKY2_GPU_BENCH_STRAIGHT").is_err() {
        return;
    }
    let empty = compile(
        "straight-line empty",
        &quotient::kernel_source_with_program(ctx.flavor, registers, nc, false, Some("")),
    );
    let straight_prologue = time(&empty, &ops_buffer, 0);
    eprintln!(
        "straight-line: prologue (0 ops) {:.3} ms",
        straight_prologue * 1e3
    );
    // Each gate on all points and on half of them: the difference prices the
    // ops per point without either kernel's fixed cost. (Repeating the ops
    // instead lets the compiler fold the straight-line repeats together.)
    let half = n as u32 / 2;
    let (mut sum_interpreter, mut sum_straight) = (0.0, 0.0);
    for (g, range) in gates.iter().enumerate() {
        let gate_ops = &ops[range.clone()];
        let buffer = upload_ops(gate_ops);
        let num_ops = gate_ops.len() as u32;
        let ti =
            time(&interpreter, &buffer, num_ops) - time_rows(&interpreter, &buffer, num_ops, half);
        let source = quotient::kernel_source_with_program(
            ctx.flavor,
            registers,
            nc,
            quotient::uses_qmul(gate_ops),
            Some(&emit(gate_ops)),
        );
        let label = format!("gate {g} ({} ops)", gate_ops.len());
        let pipeline = compile(&label, &source);
        let ts = time(&pipeline, &buffer, num_ops) - time_rows(&pipeline, &buffer, num_ops, half);
        sum_interpreter += ti;
        sum_straight += ts;
        eprintln!(
            "{label}: per half of the points: interpreter {:8.3} ms ({:.1} ps/point/op), straight-line {:8.3} ms ({:.1} ps/point/op), ratio {:.2}",
            ti * 1e3,
            ti / half as f64 / gate_ops.len() as f64 * 1e12,
            ts * 1e3,
            ts / half as f64 / gate_ops.len() as f64 * 1e12,
            ti / ts
        );
    }
    eprintln!(
        "all gates, half of the points: interpreter {:.3} ms, straight-line {:.3} ms, ratio {:.2}",
        sum_interpreter * 1e3,
        sum_straight * 1e3,
        sum_interpreter / sum_straight
    );
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

/// Occupancy probes for the quotient interpreter on a dumped program
/// (`PLONKY2_GPU_PROGRAM`): the kernel at several workgroup sizes, and, for a
/// program without quintic ops (dumped from a build whose `compile_program`
/// does not fuse them), the kernel with and without the unused quintic case
/// compiled in, which isolates that case's register cost. Blocks alternate,
/// best of five each.
/// Run alone: `PLONKY2_GPU_PROGRAM=... cargo test --release --features
/// gpu-profiling bench_quotient_occupancy -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_quotient_occupancy() {
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
    let ops = lines
        .map(|op| [op[0], op[1], op[2], op[3]])
        .collect::<Vec<_>>();
    let (n, n_lde) = (params[0] as usize, params[2] as usize);
    let (cs_cols, w_cols, z_cols, chal_len, nc, registers) = (
        columns[0] as usize,
        columns[1] as usize,
        columns[2] as usize,
        columns[3] as usize,
        columns[4],
        columns[5],
    );
    let uses_qmul = quotient::uses_qmul(&ops);
    eprintln!(
        "{} ops, {registers} registers, quintic ops: {uses_qmul}",
        ops.len()
    );
    let mut rng = StdRng::seed_from_u64(0x4f43435550);
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
    let bytes = quotient::program_words(&ops)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let ops_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ops"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let layout = storage_layout(
        &ctx.device,
        "quotient occupancy",
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
    let uniform = ctx.uniform("params", &params);
    let bindings = ctx.bind(
        &layout,
        &[
            &uniform,
            &cs_lde,
            &w_lde,
            &z_lde,
            &row_table,
            &chal,
            &ops_buffer,
            &out,
        ],
    );
    // (label, workgroup size, include the quintic case)
    let mut variants = [32u32, 64, 128, 256]
        .map(|size| (format!("workgroup {size}"), size, uses_qmul))
        .to_vec();
    if !uses_qmul {
        variants.push(("workgroup 64, unused quintic case".to_string(), 64, true));
    }
    let pipelines = variants
        .iter()
        .map(|(label, size, qmul)| {
            let source = quotient::kernel_source(ctx.flavor, registers, nc, *qmul).replace(
                "@compute @workgroup_size(64)\nfn quotient(",
                &format!("@compute @workgroup_size({size})\nfn quotient("),
            );
            assert!(source.contains(&format!("@workgroup_size({size})\nfn quotient(")));
            let module = module(&ctx.device, label, &source);
            let pipeline = pipeline(&ctx.device, &layout, &module, "quotient");
            (pipeline, *size)
        })
        .collect::<Vec<_>>();
    let time = |pipeline: &wgpu::ComputePipeline, size: u32| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut encoder = ctx.encoder("quotient occupancy");
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(size), 1, 1);
            }
            let start = Instant::now();
            ctx.submit(encoder);
            best = best.min(start.elapsed().as_secs_f64());
        }
        best
    };
    // Warm every pipeline (Metal compiles at first dispatch), then alternate.
    for (pipeline, size) in &pipelines {
        time(pipeline, *size);
    }
    let mut results = vec![Vec::new(); variants.len()];
    for round in 0..4 {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..variants.len()).collect()
        } else {
            (0..variants.len()).rev().collect()
        };
        for i in order {
            results[i].push(time(&pipelines[i].0, pipelines[i].1) * 1e3);
        }
    }
    for ((label, _, _), times) in variants.iter().zip(&results) {
        let ns = times.iter().map(|t| t * 1e6 / n as f64).collect::<Vec<_>>();
        eprintln!(
            "{label:36} {} ms  ({} ns per point)",
            times
                .iter()
                .map(|t| format!("{t:7.1}"))
                .collect::<Vec<_>>()
                .join(" "),
            ns.iter()
                .map(|t| format!("{t:.0}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

/// Transfer probes for the discrete-GPU gaps of PERF-45 and PERF-46, at the
/// MainPod wires size (135 columns of 2^16 words, about 70 MB) and smaller:
/// device to readback-buffer copies (what a commitment's coefficient readback
/// costs on the queue), device to device copies for reference, and three
/// ways to fill a device buffer from the host (a fresh buffer mapped at
/// creation, as the witness gather does; a reused mapped staging buffer and a
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
