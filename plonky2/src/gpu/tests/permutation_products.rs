//! Chunk products against direct CPU ratios, including partial final chunks.
use super::*;

#[test]
fn chunk_products_match_cpu_across_shapes_and_flavors() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let degree = 256usize;
    let challenges = 2usize;
    let mut rng = StdRng::seed_from_u64(0x4348554e4b);
    let twiddles = ctx.twiddles(8);
    let points = F::two_adic_subgroup(8);
    for flavor in inversion::flavors(&ctx) {
        for (routed, max_degree) in [
            (1usize, 1usize),
            (7, 8),
            (8, 8),
            (9, 8),
            (79, 8),
            (80, 8),
            (81, 8),
            (128, 8),
            (80, 3),
            (128, 1),
            (128, 2),
        ] {
            let chunks = routed.div_ceil(max_degree);
            let mut wires = random_fields(&mut rng, routed * degree);
            // beta = gamma = 0 for the first challenge: exercise the old
            // batch-inversion behavior when a denominator is zero as well.
            wires[0] = F::ZERO;
            let sigmas = random_fields(&mut rng, routed * degree);
            let mut constants = random_fields(&mut rng, challenges * (2 + routed));
            constants[0] = F::ZERO;
            constants[1] = F::ZERO;
            let source = if max_degree <= 2 {
                shader::permutation_module_individual(flavor)
            } else {
                shader::permutation_module_with_chunks(flavor, chunks as u32)
            };
            let module = module(&ctx.device, "chunk products test", &source);
            let kernel = pipeline(
                &ctx.device,
                &ctx.permutation_layout,
                &module,
                "row_products",
            );
            let w = ctx.upload("test wires", &field_words(&wires));
            let s = ctx.upload("test sigmas", &field_words(&sigmas));
            let c = ctx.upload("test constants", &field_words(&constants));
            let scratch = ctx.storage(
                "test chunk products",
                (challenges * degree * chunks * 8) as u64,
            );
            let params = ctx.uniform(
                "test params",
                &[
                    degree as u32,
                    routed as u32,
                    chunks as u32,
                    max_degree as u32,
                    challenges as u32,
                    1,
                    (2 * challenges) as u32,
                    0,
                ],
            );
            let bindings =
                ctx.bind_permutation(&[&params, &w, &s, &c, &twiddles, &scratch, &ctx.dummy_rw]);
            let mut encoder = ctx.encoder("test chunk products");
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&kernel);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups((degree as u32).div_ceil(64), challenges as u32, 1);
            }
            ctx.submit(encoder);
            let actual = ctx.read_u64(&scratch, challenges * degree * chunks);
            for challenge in 0..challenges {
                let beta = constants[2 * challenge];
                let gamma = constants[2 * challenge + 1];
                for row in 0..degree {
                    let denominators = (0..routed)
                        .map(|j| wires[j * degree + row] + beta * sigmas[j * degree + row] + gamma)
                        .collect::<Vec<_>>();
                    let zero = denominators.contains(&F::ZERO);
                    for chunk in 0..chunks {
                        let expected = if zero {
                            F::ZERO
                        } else {
                            (chunk * max_degree..((chunk + 1) * max_degree).min(routed))
                                .map(|j| {
                                    let n = wires[j * degree + row]
                                        + constants[2 * challenges + challenge * routed + j]
                                            * points[row]
                                        + gamma;
                                    n / denominators[j]
                                })
                                .product::<F>()
                        };
                        assert_eq!(
                            F::from_noncanonical_u64(actual[(challenge * degree + row) * chunks + chunk]),
                            expected,
                            "{flavor:?}, routed={routed}, max_degree={max_degree}, challenge={challenge}, row={row}, chunk={chunk}"
                        );
                    }
                }
            }
        }
    }
}
