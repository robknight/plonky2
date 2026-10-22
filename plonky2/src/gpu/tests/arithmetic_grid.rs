//! VAL-05: constructed worst-case inputs for the shader field arithmetic.
//! Random and fixed boundary values can miss reduction corner cases (a
//! reducer that is wrong only for products such as 2^48 * 2^48 = 2^96), so
//! every pair from a grid of powers of two and values near 2^32, p and 2^64
//! goes through multiplication, squaring, addition, subtraction,
//! multiply-add, multiply-subtract and the Poseidon S-box, on both
//! flavours, against the CPU.
use super::inversion::flavors;
use super::*;

const OPS: usize = 8;

fn grid() -> Vec<u64> {
    let mut words = (0..64).map(|i| 1u64 << i).collect::<Vec<_>>();
    for base in [1u64 << 32, GOLDILOCKS_ORDER, 0] {
        for delta in 0..4u64 {
            words.push(base.wrapping_add(delta));
            words.push(base.wrapping_sub(delta + 1));
        }
    }
    words.extend([
        0xffff_ffff_0000_0000,
        0x0000_0000_ffff_ffff,
        0xffff_ffff_ffff_ffff,
        0x8000_0000_8000_0000,
        (1 << 48) - 1,
        (1 << 48) + 1,
    ]);
    words.extend(boundary_words());
    words.sort_unstable();
    words.dedup();
    words
}

fn source(flavor: Flavor) -> String {
    let mut source = shader::prelude(flavor);
    source.push_str(
        r#"
struct Params { count: u32, n: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<Field>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Field>;
@compute @workgroup_size(64)
fn grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= params.count) { return; }
    let a = inputs[k / params.n];
    let b = inputs[k % params.n];
    outputs[8u * k] = canonical(mul_field(a, b));
    outputs[8u * k + 1u] = canonical(square_field(a));
    outputs[8u * k + 2u] = canonical(add_field(a, b));
    outputs[8u * k + 3u] = canonical(sub_field(a, b));
    outputs[8u * k + 4u] = canonical(mul_add(a, a, b));
    outputs[8u * k + 5u] = canonical(sbox(a));
    outputs[8u * k + 6u] = canonical(mul_sub(a, a, b));
    outputs[8u * k + 7u] = canonical(mul_rsub(a, a, b));
}
"#,
    );
    source
}

#[test]
fn field_arithmetic_matches_cpu_on_worst_case_grid() {
    let _guard = ENABLE_LOCK.lock().unwrap();
    let Some(ctx) = ctx() else { return };
    let words = grid();
    let n = words.len();
    let count = n * n;
    let expected = (0..count)
        .flat_map(|k| {
            let a = F::from_noncanonical_u64(words[k / n]);
            let b = F::from_noncanonical_u64(words[k % n]);
            let sbox = a.exp_u64(7);
            [
                a * b,
                a * a,
                a + b,
                a - b,
                a + a * b,
                sbox,
                a - a * b,
                a * b - a,
            ]
            .map(|x| x.to_canonical_u64())
        })
        .collect::<Vec<_>>();
    let input = ctx.upload("grid inputs", &words);
    let output = ctx.storage("grid outputs", (count * OPS * 8) as u64);
    let params = ctx.uniform("grid params", &[count as u32, n as u32, 0, 0]);
    for flavor in flavors(&ctx) {
        let layout = storage_layout(
            &ctx.device,
            "grid",
            &[(true, true), (false, true), (false, false)],
        );
        let module = module(&ctx.device, "grid", &source(flavor));
        let pipeline = pipeline(&ctx.device, &layout, &module, "grid");
        let bindings = ctx.bind(&layout, &[&params, &input, &output]);
        let mut encoder = ctx.encoder("grid");
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        ctx.submit(encoder);
        let actual = ctx.read(&output, (count * OPS * 8) as u64, |bytes| {
            bytes
                .chunks_exact(8)
                .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                .collect::<Vec<_>>()
        });
        for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
            let (k, op) = (i / OPS, i % OPS);
            assert_eq!(
                got,
                want,
                "{flavor:?} op {op} on {:#x}, {:#x}",
                words[k / n],
                words[k % n]
            );
        }
    }
}
