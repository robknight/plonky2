//! Exact checks and throughput comparisons of addition-chain and binary field inversion.
use super::*;

struct Probe {
    layout: wgpu::BindGroupLayout,
    binary: wgpu::ComputePipeline,
    chain: wgpu::ComputePipeline,
}

impl Probe {
    fn new(ctx: &GpuContext, flavor: Flavor) -> Self {
        let mut source = shader::prelude(flavor);
        source.push_str(
            r#"
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<Field>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Field>;
// Binary exponentiation by p - 2, used as a test and benchmark reference.
fn inv_binary_reference(a: Field) -> Field {
    var result = a;
    var base = a;
    for (var i = 1u; i < 64u; i++) {
        base = square_field(base);
        if (i != 32u) { result = mul_field(result, base); }
    }
    return result;
}
@compute @workgroup_size(64)
fn inverse_binary(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.count) { outputs[id.x] = canonical(inv_binary_reference(inputs[id.x])); }
}
@compute @workgroup_size(64)
fn inverse_chain(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.count) { outputs[id.x] = canonical(inv_field(inputs[id.x])); }
}
"#,
        );
        let layout = storage_layout(
            &ctx.device,
            "inverse probe",
            &[(true, true), (false, true), (false, false)],
        );
        let module = module(&ctx.device, "inverse probe", &source);
        Self {
            binary: pipeline(&ctx.device, &layout, &module, "inverse_binary"),
            chain: pipeline(&ctx.device, &layout, &module, "inverse_chain"),
            layout,
        }
    }

    fn encode(
        &self,
        ctx: &GpuContext,
        bindings: &wgpu::BindGroup,
        count: usize,
        chain: bool,
    ) -> wgpu::CommandEncoder {
        let mut encoder = ctx.encoder("inverse probe");
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(if chain { &self.chain } else { &self.binary });
            pass.set_bind_group(0, bindings, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        encoder
    }
}

fn inputs(count: usize) -> Vec<u64> {
    let mut words = boundary_words();
    for delta in 0..32 {
        words.extend([
            GOLDILOCKS_ORDER - 1 - delta,
            GOLDILOCKS_ORDER + delta,
            (1u64 << 32) - 1 - delta,
            (1u64 << 32) + delta,
        ]);
    }
    let mut rng = StdRng::seed_from_u64(0x494e5645525345);
    words.resize_with(count, || rng.next_u64());
    words
}

fn expected(words: &[u64]) -> Vec<u64> {
    words
        .iter()
        .map(|&word| {
            F::from_noncanonical_u64(word)
                .try_inverse()
                .unwrap_or(F::ZERO)
                .to_canonical_u64()
        })
        .collect()
}

fn read(ctx: &GpuContext, output: &wgpu::Buffer, count: usize) -> Vec<u64> {
    ctx.read(output, count as u64 * 8, |bytes| {
        bytes
            .chunks_exact(8)
            .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
            .collect()
    })
}

pub(super) fn flavors(ctx: &GpuContext) -> Vec<Flavor> {
    let mut flavors = vec![Flavor::Portable];
    if ctx.device.features().contains(wgpu::Features::SHADER_INT64) {
        flavors.push(Flavor::Native);
    }
    flavors
}

#[test]
fn inverse_chain_matches_cpu_and_previous_inverse() {
    let ctx = GpuContext::new().expect("inverse regression requires a physical GPU");
    eprintln!("INVERSION_DEVICE {ctx:?}");
    let words = inputs(8209); // Uneven final workgroup, canonical and noncanonical inputs.
    let expected = expected(&words);
    let source = ctx.upload("inverse test input", &words);
    let output = ctx.storage("inverse test output", words.len() as u64 * 8);
    let params = ctx.uniform("inverse test params", &[words.len() as u32, 0, 0, 0]);
    for flavor in flavors(&ctx) {
        let probe = Probe::new(&ctx, flavor);
        let bindings = ctx.bind(&probe.layout, &[&params, &source, &output]);
        for chain in [false, true] {
            ctx.submit(probe.encode(&ctx, &bindings, words.len(), chain));
            assert_eq!(
                read(&ctx, &output, words.len()),
                expected,
                "{flavor:?}, chain={chain}"
            );
        }
    }
}

#[test]
#[ignore = "explicit addition-chain versus binary inverse throughput experiment"]
fn measure_inverse_chain() {
    use std::time::Instant;
    let ctx = GpuContext::new().expect("inverse benchmark requires a physical GPU");
    eprintln!("INVERSION_DEVICE {ctx:?}");
    for flavor in flavors(&ctx) {
        let probe = Probe::new(&ctx, flavor);
        for count in [4096usize, 65536] {
            let words = inputs(count);
            let expected = expected(&words);
            let source = ctx.upload("inverse benchmark input", &words);
            let output = ctx.storage("inverse benchmark output", count as u64 * 8);
            let params = ctx.uniform("inverse benchmark params", &[count as u32, 0, 0, 0]);
            let bindings = ctx.bind(&probe.layout, &[&params, &source, &output]);
            // Two warmup rounds; then four ABBA blocks, eight samples per variant.
            for round in 0..10 {
                let order = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for chain in order {
                    let encoder = probe.encode(&ctx, &bindings, count, chain);
                    let start = Instant::now();
                    ctx.submit(encoder);
                    let elapsed = start.elapsed().as_secs_f64() * 1e6;
                    assert_eq!(read(&ctx, &output, count), expected);
                    eprintln!("INVERSION_BENCH flavor={flavor:?} count={count} round={round} chain={chain} submit_wait_us={elapsed:.3}");
                }
            }
        }
    }
}
