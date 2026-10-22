//! Compare every emitted wire against an extension-field reference implementation.
//! MODE bits isolate the base-field S-box, MDS and fast partial MDS routines.
use std::hint::black_box;
use std::time::Instant;

use anyhow::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::field::goldilocks_field::GoldilocksField as F;
use crate::field::types::{Field, Field64, Sample};
use crate::gates::poseidon::{PoseidonGate, PoseidonGenerator};
use crate::hash::poseidon::{self, Poseidon, SPONGE_WIDTH};
use crate::iop::generator::{GeneratedValues, SimpleGenerator};
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
type Gate = PoseidonGate<F, 2>;

fn inputs<'a>(
    representatives: &'a [usize],
    state: [F; SPONGE_WIDTH],
    swap: bool,
) -> PartitionWitness<'a, F> {
    let mut witness = PartitionWitness::new(135, 1, representatives);
    for (column, value) in state.into_iter().enumerate() {
        witness.set_wire(Wire { row: 0, column }, value).unwrap();
    }
    witness
        .set_wire(
            Wire {
                row: 0,
                column: Gate::WIRE_SWAP,
            },
            F::from_bool(swap),
        )
        .unwrap();
    witness
}

#[test]
fn intermediate_wires_match_generic_reference() {
    let representatives: Vec<_> = (0..135).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(0x504f534549444f4e);
    let mut cases = vec![
        [F::ZERO; SPONGE_WIDTH],
        [F::ONE; SPONGE_WIDTH],
        [F::NEG_ONE; SPONGE_WIDTH],
        [F::from_noncanonical_u64(u64::MAX); SPONGE_WIDTH],
        std::array::from_fn(|i| {
            F::from_noncanonical_u64([0, 1, F::ORDER - 1, F::ORDER, F::ORDER + 1, u64::MAX][i % 6])
        }),
    ];
    cases.extend((0..128).map(|_| std::array::from_fn(|_| F::sample(&mut rng))));
    let generator = PoseidonGenerator::<F, 2>::default();
    for state in cases {
        for swap in [false, true] {
            let witness = inputs(&representatives, state, swap);
            let mut expected = GeneratedValues::empty();
            reference::<0>(&witness, &mut expected).unwrap();
            assert_eq!(expected.target_values.len(), Gate::end() - SPONGE_WIDTH - 1);
            for run in [
                reference::<1>,
                reference::<2>,
                reference::<4>,
                reference::<7>,
            ] {
                let mut actual = GeneratedValues::empty();
                run(&witness, &mut actual).unwrap();
                assert_eq!(actual.target_values, expected.target_values);
            }
            let mut actual = GeneratedValues::empty();
            generator.run_once(&witness, &mut actual).unwrap();
            assert_eq!(actual.target_values, expected.target_values);
        }
    }
}

#[test]
#[ignore]
fn benchmark_generator_layers() {
    let representatives: Vec<_> = (0..135).collect();
    let state = std::array::from_fn(|i| F::from_canonical_usize(i * 7919 + 1));
    let witness = inputs(&representatives, state, true);
    type Runner = fn(&PartitionWitness<F>, &mut GeneratedValues<F>) -> Result<()>;
    let variants: [(&str, Runner); 5] = [
        ("generic", reference::<0>),
        ("neon-sbox", reference::<1>),
        ("neon-mds", reference::<2>),
        ("base-partial", reference::<4>),
        ("all-base", reference::<7>),
    ];
    let mut out = GeneratedValues::with_capacity(Gate::end() - SPONGE_WIDTH - 1);
    for epoch in 0..8 {
        let order: Vec<_> = if epoch % 2 == 0 {
            (0..5).collect()
        } else {
            (0..5).rev().collect()
        };
        for index in order {
            let (name, run) = variants[index];
            for _ in 0..1000 {
                out.target_values.clear();
                run(black_box(&witness), black_box(&mut out)).unwrap();
                black_box(&out);
            }
            let start = Instant::now();
            for _ in 0..20000 {
                out.target_values.clear();
                run(black_box(&witness), black_box(&mut out)).unwrap();
                black_box(&out);
            }
            println!(
                "layer-bench mode={name} ns={:.3}",
                start.elapsed().as_nanos() as f64 / 20000.0
            );
        }
    }
}

fn reference<const MODE: u8>(
    witness: &PartitionWitness<F>,
    out_buffer: &mut GeneratedValues<F>,
) -> Result<()> {
    let local_wire = |column| Wire { row: 0, column };

    let mut state = (0..SPONGE_WIDTH)
        .map(|i| witness.get_wire(local_wire(Gate::wire_input(i))))
        .collect::<Vec<_>>();

    let swap_value = witness.get_wire(local_wire(Gate::WIRE_SWAP));
    debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

    for i in 0..4 {
        let delta_i = swap_value * (state[i + 4] - state[i]);
        out_buffer.set_wire(local_wire(Gate::wire_delta(i)), delta_i)?;
    }

    if swap_value == F::ONE {
        for i in 0..4 {
            state.swap(i, 4 + i);
        }
    }

    let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();
    let mut round_ctr = 0;

    for r in 0..poseidon::HALF_N_FULL_ROUNDS {
        <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
        if r != 0 {
            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(local_wire(Gate::wire_full_sbox_0(r, i)), state[i])?;
            }
        }
        if MODE & 1 != 0 {
            F::sbox_layer(&mut state);
        } else {
            F::sbox_layer_field(&mut state);
        }
        state = if MODE & 2 != 0 {
            F::mds_layer(&state)
        } else {
            F::mds_layer_field(&state)
        };
        round_ctr += 1;
    }

    <F as Poseidon>::partial_first_constant_layer(&mut state);
    state = <F as Poseidon>::mds_partial_layer_init(&state);
    for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
        out_buffer.set_wire(local_wire(Gate::wire_partial_sbox(r)), state[0])?;
        state[0] = <F as Poseidon>::sbox_monomial(state[0]);
        state[0] += F::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
        state = if MODE & 4 != 0 {
            F::mds_partial_layer_fast(&state, r)
        } else {
            F::mds_partial_layer_fast_field(&state, r)
        };
    }
    out_buffer.set_wire(
        local_wire(Gate::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)),
        state[0],
    )?;
    state[0] = <F as Poseidon>::sbox_monomial(state[0]);
    state = if MODE & 4 != 0 {
        F::mds_partial_layer_fast(&state, poseidon::N_PARTIAL_ROUNDS - 1)
    } else {
        F::mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1)
    };
    round_ctr += poseidon::N_PARTIAL_ROUNDS;

    for r in 0..poseidon::HALF_N_FULL_ROUNDS {
        <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(Gate::wire_full_sbox_1(r, i)), state[i])?;
        }
        if MODE & 1 != 0 {
            F::sbox_layer(&mut state);
        } else {
            F::sbox_layer_field(&mut state);
        }
        state = if MODE & 2 != 0 {
            F::mds_layer(&state)
        } else {
            F::mds_layer_field(&state)
        };
        round_ctr += 1;
    }

    for i in 0..SPONGE_WIDTH {
        out_buffer.set_wire(local_wire(Gate::wire_output(i)), state[i])?
    }

    Ok(())
}

// Experimental batching only: each lane is an independent witness. The runner
// still invokes scalar generators; this measures packing and scatter overhead too.
fn packed_reference<P: crate::field::packed::PackedField<Scalar = F>>(
    witnesses: &[PartitionWitness<F>],
    outputs: &mut [GeneratedValues<F>],
) -> Result<()> {
    assert_eq!(witnesses.len(), P::WIDTH);
    assert_eq!(outputs.len(), P::WIDTH);
    assert!(P::WIDTH <= 8);
    let pack_wire = |column| {
        let mut lanes = [F::ZERO; 8];
        for lane in 0..P::WIDTH {
            lanes[lane] = witnesses[lane].get_wire(Wire { row: 0, column });
        }
        *P::from_slice(&lanes[..P::WIDTH])
    };
    let emit = |column, value: P, outputs: &mut [GeneratedValues<F>]| -> Result<()> {
        for (output, &lane) in outputs.iter_mut().zip(value.as_slice()) {
            output.set_wire(Wire { row: 0, column }, lane)?;
        }
        Ok(())
    };
    let mut state: [P; SPONGE_WIDTH] = std::array::from_fn(pack_wire);
    let swap = pack_wire(Gate::WIRE_SWAP);
    for i in 0..4 {
        let delta = swap * (state[i + 4] - state[i]);
        emit(Gate::wire_delta(i), delta, outputs)?;
        state[i] += delta;
        state[i + 4] -= delta;
    }
    let sbox = |x: P| {
        let x2 = x.square();
        let x4 = x2.square();
        (x * x2) * x4
    };
    let mut round = 0;
    for r in 0..poseidon::HALF_N_FULL_ROUNDS {
        for i in 0..SPONGE_WIDTH {
            state[i] +=
                F::from_canonical_u64(poseidon::ALL_ROUND_CONSTANTS[round * SPONGE_WIDTH + i]);
            if r != 0 {
                emit(Gate::wire_full_sbox_0(r, i), state[i], outputs)?;
            }
            state[i] = sbox(state[i]);
        }
        state = F::mds_layer_packed_field::<F, 2, F, P, 1>(&state);
        round += 1;
    }
    for i in 0..SPONGE_WIDTH {
        state[i] += F::from_canonical_u64(F::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
    }
    state = F::mds_partial_layer_init_packed_field::<F, 2, F, P, 1>(&state);
    for r in 0..poseidon::N_PARTIAL_ROUNDS {
        emit(Gate::wire_partial_sbox(r), state[0], outputs)?;
        state[0] = sbox(state[0]);
        if r + 1 != poseidon::N_PARTIAL_ROUNDS {
            state[0] += F::from_canonical_u64(F::FAST_PARTIAL_ROUND_CONSTANTS[r]);
        }
        state = F::mds_partial_layer_fast_packed_field::<F, 2, F, P, 1>(&state, r);
    }
    round += poseidon::N_PARTIAL_ROUNDS;
    for r in 0..poseidon::HALF_N_FULL_ROUNDS {
        for i in 0..SPONGE_WIDTH {
            state[i] +=
                F::from_canonical_u64(poseidon::ALL_ROUND_CONSTANTS[round * SPONGE_WIDTH + i]);
            emit(Gate::wire_full_sbox_1(r, i), state[i], outputs)?;
            state[i] = sbox(state[i]);
        }
        state = F::mds_layer_packed_field::<F, 2, F, P, 1>(&state);
        round += 1;
    }
    for i in 0..SPONGE_WIDTH {
        emit(Gate::wire_output(i), state[i], outputs)?;
    }
    Ok(())
}

#[test]
fn packed_intermediate_wires_match() {
    use crate::field::packable::Packable;
    use crate::field::packed::PackedField;
    type P = <F as Packable>::Packing;
    let representatives: Vec<_> = (0..Gate::end()).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(0x53494d44504f5345);
    for case in 0..133 {
        let witnesses: Vec<_> = (0..P::WIDTH)
            .map(|lane| {
                let state = std::array::from_fn(|i| {
                    if case < 5 {
                        F::from_noncanonical_u64(
                            [0, 1, F::ORDER - 1, F::ORDER, u64::MAX][(case + lane + i) % 5],
                        )
                    } else {
                        F::sample(&mut rng)
                    }
                });
                inputs(&representatives, state, (lane + case) % 2 == 0)
            })
            .collect();
        let mut outputs: Vec<_> = (0..P::WIDTH).map(|_| GeneratedValues::empty()).collect();
        packed_reference::<P>(&witnesses, &mut outputs).unwrap();
        for (witness, actual) in witnesses.iter().zip(outputs) {
            let mut expected = GeneratedValues::empty();
            reference::<0>(witness, &mut expected).unwrap();
            assert_eq!(actual.target_values, expected.target_values);
        }
    }
    println!(
        "packed-correctness width={} batches=133 wires_per_lane=122",
        P::WIDTH
    );
}

#[test]
#[ignore]
fn benchmark_packed_generator() {
    use crate::field::packable::Packable;
    use crate::field::packed::PackedField;
    type P = <F as Packable>::Packing;
    let representatives: Vec<_> = (0..Gate::end()).collect();
    let witnesses: Vec<_> = (0..P::WIDTH)
        .map(|lane| {
            inputs(
                &representatives,
                std::array::from_fn(|i| F::from_canonical_usize(i * 7919 + lane * 101 + 1)),
                lane % 2 == 0,
            )
        })
        .collect();
    let mut outputs: Vec<_> = (0..P::WIDTH)
        .map(|_| GeneratedValues::with_capacity(122))
        .collect();
    let run = |packed: bool, outputs: &mut [GeneratedValues<F>]| {
        for out in outputs.iter_mut() {
            out.target_values.clear();
        }
        if packed {
            packed_reference::<P>(black_box(&witnesses), black_box(outputs)).unwrap();
        } else {
            for (witness, out) in witnesses.iter().zip(outputs.iter_mut()) {
                reference::<7>(black_box(witness), black_box(out)).unwrap();
            }
        }
    };
    for epoch in 0..8 {
        let order = if epoch % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for packed in order {
            for _ in 0..1000 {
                run(packed, &mut outputs);
                black_box(&outputs);
            }
            let start = Instant::now();
            for _ in 0..20000 {
                run(packed, &mut outputs);
                black_box(&outputs);
            }
            println!(
                "packed-bench packed={packed} width={} ns_per_generator={:.3}",
                P::WIDTH,
                start.elapsed().as_nanos() as f64 / (20000 * P::WIDTH) as f64
            );
        }
    }
}
