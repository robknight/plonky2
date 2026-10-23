//! Larger NTT tiles preserve forward, padded and inverse transforms.
use super::*;
use crate::gpu::lde::{fft_groups_with_tile, FftJob};

#[test]
fn large_tiles_match_cpu_fft_and_coset_ifft() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    if ctx.device.limits().max_compute_workgroup_storage_size < 16384 {
        return;
    }
    let mut rng = StdRng::seed_from_u64(0x54494c45);
    for log_n in [10u32, 11, 12, 18, 19] {
        let n = 1usize << log_n;
        for inverse in [false, true] {
            let len = if inverse { n } else { n / 8 };
            let mut input = random_columns(&mut rng, 2, len);
            input[0][0] = F::from_noncanonical_u64(u64::MAX);
            let shift = F::coset_shift();
            let expected = input
                .iter()
                .map(|column| {
                    if inverse {
                        PolynomialValues::new(column.clone())
                            .coset_ifft(shift)
                            .coeffs
                    } else {
                        PolynomialCoeffs::new(column.clone())
                            .lde(3)
                            .coset_fft(shift)
                            .values
                    }
                })
                .collect::<Vec<_>>();
            let columns = input.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let values = lde::upload_columns(&ctx, "fft tile test input", &columns);
            let data = ctx.storage("fft tile test data", (2 * n * 8) as u64);
            let out = ctx.storage("fft tile test output", (2 * n * 8) as u64);
            for bits in [10, 11] {
                let mut encoder = ctx.encoder("fft tile test");
                fft_groups_with_tile(
                    &ctx,
                    &mut encoder,
                    FftJob {
                        log_n,
                        columns: 2,
                        input: &values,
                        in_len: len as u32,
                        in_stride: len as u32,
                        data: &data,
                        output: inverse.then_some(&*out),
                        shift_in: (!inverse).then_some(shift),
                        inverse,
                        shift_out: inverse.then(|| shift.inverse()),
                    },
                    bits,
                    None,
                    None,
                );
                ctx.submit(encoder);
                let actual = lde::read_columns::<F>(&ctx, if inverse { &out } else { &data }, 2, n);
                assert_eq!(
                    actual, expected,
                    "bits={bits}, log_n={log_n}, inverse={inverse}"
                );
            }
        }
    }
}
