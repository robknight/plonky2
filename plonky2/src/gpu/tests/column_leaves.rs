//! Column storage preserves sparse queries, CPU fallback and public serialization.
use super::*;

#[test]
fn column_leaves_match_rows_through_queries_and_materialization() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    for staging in [false, true] {
        let Ok(mut context) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        if staging {
            context.mappable = false;
        }
        let ctx = Arc::new(context);
        for (columns, log_n, arity_bits) in [
            (1usize, 0u32, 0u32),
            (13, 10, 0),
            (2, 10, 4),
            (139, 10, 0),
            (2, 10, 10),
        ] {
            let n = 1usize << log_n;
            let words = (0..columns * n)
                .map(|i| (i as u64).wrapping_mul(0xfedc_ba98_7654_3211))
                .collect::<Vec<_>>();
            let leaf_len = columns << arity_bits;
            let count = n >> arity_bits;
            let expected = (0..count)
                .flat_map(|i| {
                    let words = &words;
                    (0..leaf_len).map(move |e| {
                        let row = crate::util::reverse_bits(
                            (i << arity_bits) + e / columns,
                            log_n as usize,
                        );
                        F::from_noncanonical_u64(words[(e % columns) * n + row])
                    })
                })
                .collect::<Vec<_>>();
            let make = || {
                let buffer = ctx.upload("column leaves test", &words);
                let done = ctx.submit_async(ctx.encoder("column leaves ready"));
                MerkleLeaves::<F>::from_device_columns(
                    ctx.clone(),
                    buffer,
                    columns,
                    log_n,
                    arity_bits,
                    done,
                )
            };
            let leaves = make();
            let mut clone = leaves.clone();
            assert!(leaves.is_column_backed());
            leaves.prepare_queries(&[]);
            let indices = [0, count / 2, count - 1, 0];
            leaves.prepare_queries(&indices);
            for i in indices {
                assert_eq!(leaves.row(i), &expected[i * leaf_len..(i + 1) * leaf_len]);
            }
            assert!(
                leaves.is_column_backed(),
                "queries must not download all columns"
            );
            if ctx.mappable && count > 3 {
                leaves.prepare_rows(&[3, 1, 3]);
                assert_eq!(leaves.row(3), &expected[3 * leaf_len..4 * leaf_len]);
                assert!(leaves.is_column_backed());
            }
            // Full fallback may run inside Rayon while another clone reads.
            let idle_before: u64 = ctx
                .readback_pool
                .lock()
                .unwrap()
                .iter()
                .map(wgpu::Buffer::size)
                .sum();
            plonky2_maybe_rayon::join(
                || assert_eq!(leaves.as_flat_slice(), expected),
                || {
                    assert_eq!(clone.as_flat_slice(), expected);
                },
            );
            let idle_after: u64 = ctx
                .readback_pool
                .lock()
                .unwrap()
                .iter()
                .map(wgpu::Buffer::size)
                .sum();
            assert!(
                idle_after <= idle_before,
                "full snapshots must not inflate retained readback memory"
            );
            assert!(!leaves.is_device_resident());
            assert!(!clone.is_device_resident());
            clone.as_flat_slice_mut()[0] += F::ONE;
            assert_eq!(leaves.row(0)[0], expected[0]);
            assert_eq!(clone.row(0)[0], expected[0] + F::ONE);
            assert_eq!(leaves.into_flat_vec(), expected);
            let fresh = make();
            let host = MerkleLeaves::new(expected, leaf_len, count);
            assert_eq!(
                serde_cbor::to_vec(&fresh).unwrap(),
                serde_cbor::to_vec(&host).unwrap()
            );
            assert!(!fresh.is_device_resident());
        }
    }
}

#[test]
fn resident_commitments_reuse_columns_while_hybrid_keeps_rows() {
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let values = (0..13)
        .map(|j| {
            PolynomialValues::new(
                (0..256)
                    .map(|i| F::from_canonical_usize(19 * i + j))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let commit = |storage| {
        lde::from_values_with_residency::<F, C, 2>(
            &ctx,
            values.clone(),
            2,
            false,
            2,
            &mut TimingTree::default(),
            storage,
            false,
        )
    };
    let rows = commit(policy::LeafStorage::Rows);
    let columns = commit(policy::LeafStorage::Columns);
    assert!(!rows.batch.merkle_tree.leaves.is_column_backed());
    assert!(columns.batch.merkle_tree.leaves.is_column_backed());
    assert_eq!(columns.batch.merkle_tree.cap, rows.batch.merkle_tree.cap);
    // Consuming the quotient's LDE cache leaves its query storage intact.
    drop(columns.batch.device.take());
    let indices = [0, 1023, 37, 37];
    columns.batch.merkle_tree.leaves.prepare_queries(&indices);
    for i in indices {
        assert_eq!(
            columns.batch.merkle_tree.leaves.row(i),
            rows.batch.merkle_tree.leaves.row(i)
        );
    }
    assert!(columns.batch.merkle_tree.leaves.is_column_backed());
    assert_eq!(columns.polynomials(), rows.polynomials());
}

#[test]
fn column_materialization_allows_concurrent_gpu_readers() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let log_n = 18u32;
    let n = 1usize << log_n;
    let columns = 13usize;
    let words = (0..n * columns).map(|i| i as u64).collect::<Vec<_>>();
    let source = ctx.upload("shared LDE test", &words);
    let target = ctx.storage("concurrent LDE reader", source.size());
    let ready = ctx.submit_async(ctx.encoder("shared LDE ready"));
    let leaves = MerkleLeaves::<F>::from_device_columns(
        ctx.clone(),
        source.clone(),
        columns,
        log_n,
        0,
        ready,
    );
    let start = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            start.wait();
            for _ in 0..32 {
                let mut encoder = ctx.encoder("concurrent LDE read");
                encoder.copy_buffer_to_buffer(&source, 0, &target, 0, source.size());
                ctx.submit(encoder);
            }
        });
        start.wait();
        let actual = leaves.as_flat_slice();
        for (i, row) in actual.chunks_exact(columns).enumerate() {
            let reversed = crate::util::reverse_bits(i, log_n as usize);
            for (j, value) in row.iter().enumerate() {
                assert_eq!(*value, F::from_canonical_usize(j * n + reversed));
            }
        }
        reader.join().unwrap();
    });
    assert!(!leaves.is_device_resident());
}
