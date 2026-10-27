//! Experimental CPU/GPU cut: only certified exclusive outputs whose consumers
//! all stay on the device may be deferred. Roots, aliases and public inputs
//! are seeded on the CPU; the optional checked cut validates them on the GPU.
//! This is selected only by the benchmark experiment hook.
use std::collections::HashMap;
use std::fmt::Write;

use anyhow::Result;

use super::cpu::Executor;
use super::{Group, Kind, RegionProgram};
use crate::field::goldilocks_field::GoldilocksField as G;
use crate::gpu::{DeviceBuffer, DeviceCache, GpuContext};
use crate::hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS};

#[derive(Debug)]
pub(crate) struct Tail {
    masks: HashMap<u32, Vec<bool>>,
    pub(super) checks: bool,
    public_slots: Vec<u32>,
    cpu_slots: Vec<u32>,
    cpu_slots_device: DeviceCache,
    metadata: Vec<u32>,
    runs: Vec<(u32, u32, bool)>,
    run_kinds: Vec<u8>,
    params_cache: Vec<[DeviceCache; 2]>,
    metadata_device: DeviceCache,
    targets_device: DeviceCache,
    coefficients_device: DeviceCache,
    pub gpu_random: Vec<bool>,
    shared_random: bool,
}
impl Tail {
    #[cfg(test)]
    pub(super) fn compile(p: &RegionProgram, public_slots: &[u32]) -> Self {
        Self::compile_with_checks(p, public_slots, false)
    }
    #[cfg(test)]
    pub(super) fn compile_with_checks(
        p: &RegionProgram,
        public_slots: &[u32],
        checks: bool,
    ) -> Self {
        Self::compile_config(p, public_slots, checks, false)
    }
    pub(super) fn compile_config(
        p: &RegionProgram,
        public_slots: &[u32],
        checks: bool,
        compress: bool,
    ) -> Self {
        #[cfg(feature = "witness-benchmarks")]
        if std::env::var_os("PLONKY2_GPU_WITNESS_CUT_ANALYSIS").is_some() {
            analyze(p, public_slots);
        }
        let fast = p.fast.get().expect("certified tail");
        let mut cpu = vec![false; p.state_slots as usize];
        cpu[..p.roots.len()].fill(true);
        for &s in public_slots {
            if s != u32::MAX {
                cpu[s as usize] = true;
            }
        }
        if !checks {
            for i in 0..p.alias_count as usize {
                cpu[p.targets[p.alias_pair_offset as usize + 2 * i] as usize] = true;
                cpu[p.targets[p.alias_pair_offset as usize + 2 * i + 1] as usize] = true;
            }
        }
        let mut masks = HashMap::new();
        let target = |g: &Group, pos: u32, k: u32| {
            p.targets[(g.targets_offset + pos * g.count + k) as usize]
        };
        let mut counts = [0usize; Kind::COUNT];
        let mut gpu_random = vec![false; p.random_count as usize];
        for groups in p.groups.iter().rev() {
            for g in groups {
                let supported = matches!(
                    g.kind,
                    Kind::Poseidon
                        | Kind::Arithmetic
                        | Kind::Copy
                        | Kind::Constant
                        | Kind::Equality
                        | Kind::ExtensionArithmetic
                        | Kind::ExtensionMultiply
                        | Kind::ArithmeticChain
                        | Kind::RandomValue
                        | Kind::ConditionalZero
                        | Kind::BitSum
                        | Kind::RandomAccess
                );
                let mut mask = vec![false; g.count as usize];
                for k in 0..g.count {
                    let offload = supported
                        && (g.inputs..g.inputs + g.outputs).all(|pos| {
                            let s = target(g, pos, k) as usize;
                            (checks && s < p.roots.len())
                                || (!cpu[s] && s >= p.roots.len() && fast.single[s] != 0)
                        });
                    mask[k as usize] = offload;
                    if offload {
                        counts[g.kind.code()] += 1;
                        if g.kind == Kind::RandomValue {
                            gpu_random[p.coefficients
                                [(g.coefficients_offset().unwrap() + k) as usize]
                                as usize] = true;
                        }
                    } else {
                        for pos in 0..g.inputs {
                            cpu[target(g, pos, k) as usize] = true;
                        }
                    }
                }
                masks.insert(g.targets_offset, mask);
            }
        }
        // A random source may have several outputs. CPU consumers keep the
        // source on the CPU, including its deferred copies on the device.
        for g in p
            .groups
            .iter()
            .flatten()
            .filter(|g| g.kind == Kind::RandomValue)
        {
            for k in 0..g.count {
                if !masks[&g.targets_offset][k as usize] {
                    gpu_random[p.coefficients[(g.coefficients_offset().unwrap() + k) as usize]
                        as usize] = false;
                }
            }
        }
        let mut shared_random = false;
        let mut levels = vec![0u32; p.state_slots as usize];
        let split_eq = crate::gpu::resident_witness::experiment().contains("split-eq");
        let mut buckets = std::collections::BTreeMap::<(u32, u8), Vec<[u32; 8]>>::new();
        for (layer, groups) in p.groups.iter().enumerate() {
            for g in groups {
                for k in 0..g.count {
                    if !masks[&g.targets_offset][k as usize] {
                        continue;
                    }
                    // Every CPU-produced value is ready before the device starts.
                    // Seeded roots are expected values, not dependencies on their
                    // deferred checking operations. Only device-produced values
                    // introduce edges in this graph.
                    let rank = if compress {
                        (0..g.inputs)
                            .map(|pos| levels[target(g, pos, k) as usize])
                            .max()
                            .unwrap_or(0)
                            + 1
                    } else {
                        layer as u32
                    };
                    if compress {
                        for pos in g.inputs..g.inputs + g.outputs {
                            let slot = target(g, pos, k) as usize;
                            if slot >= p.roots.len() {
                                levels[slot] = rank;
                            }
                        }
                    }
                    let device_random = if g.kind == Kind::RandomValue {
                        let device = gpu_random[p.coefficients
                            [(g.coefficients_offset().unwrap() + k) as usize]
                            as usize];
                        shared_random |= !device;
                        u32::from(device)
                    } else {
                        0
                    };
                    buckets
                        .entry((
                            rank,
                            if g.kind == Kind::Poseidon {
                                0
                            } else if split_eq && g.kind == Kind::Equality {
                                2
                            } else {
                                1
                            },
                        ))
                        .or_default()
                        .push([
                            g.kind.code() as u32,
                            g.targets_offset,
                            g.count,
                            k,
                            g.argument.word(),
                            g.inputs,
                            g.outputs,
                            device_random,
                        ]);
                }
            }
        }
        let mut metadata = Vec::new();
        let mut runs = Vec::new();
        let mut run_kinds = Vec::new();
        for ((_, kind), descriptors) in buckets {
            let base = (metadata.len() / 8) as u32;
            let count = descriptors.len() as u32;
            for descriptor in descriptors {
                metadata.extend(descriptor);
            }
            runs.push((base, count, kind == 0));
            run_kinds.push(kind);
        }
        eprintln!(
            "GPU_TAIL_SCHEDULE compressed={compress} dispatches={}",
            runs.len()
        );
        let mut device_outputs = vec![false; p.state_slots as usize];
        for g in p.groups.iter().flatten() {
            for k in 0..g.count {
                if masks[&g.targets_offset][k as usize] {
                    for pos in g.inputs..g.inputs + g.outputs {
                        let slot = target(g, pos, k) as usize;
                        if slot >= p.roots.len() {
                            device_outputs[slot] = true;
                        }
                    }
                }
            }
        }
        let cpu_slots = device_outputs
            .iter()
            .enumerate()
            .filter_map(|(i, &gpu)| (!gpu).then_some(i as u32))
            .collect::<Vec<_>>();
        eprintln!(
            "GPU_TAIL_STATE cpu_bytes={} total_bytes={}",
            cpu_slots.len() * 8,
            p.state_slots as usize * 8
        );
        eprintln!(
            "GPU_TAIL_PLAN counts={counts:?} operations={} dispatches={} gpu_random={}/{}",
            metadata.len() / 8,
            runs.len(),
            gpu_random.iter().filter(|&&x| x).count(),
            p.random_count
        );
        Self {
            masks,
            checks,
            public_slots: public_slots.to_vec(),
            cpu_slots,
            cpu_slots_device: Default::default(),
            metadata,
            params_cache: runs
                .iter()
                .map(|_| [Default::default(), Default::default()])
                .collect(),
            runs,
            run_kinds,
            metadata_device: Default::default(),
            targets_device: Default::default(),
            coefficients_device: Default::default(),
            gpu_random,
            shared_random,
        }
    }
    pub(super) fn matches_public_inputs(&self, slots: &[u32]) -> bool {
        self.public_slots == slots
    }
    pub(super) fn cpu_ranges(
        &self,
        g: &Group,
        start: u32,
        end: u32,
        mut run: impl FnMut(u32, u32) -> Result<()>,
    ) -> Result<()> {
        let mask = &self.masks[&g.targets_offset];
        let mut i = start;
        while i < end {
            while i < end && mask[i as usize] {
                i += 1;
            }
            let begin = i;
            while i < end && !mask[i as usize] {
                i += 1;
            }
            if begin < i {
                run(begin, i)?;
            }
        }
        Ok(())
    }
    pub(super) fn upload_state(&self, e: &Executor<'_>, ctx: &GpuContext) -> DeviceBuffer {
        use std::sync::atomic::Ordering;

        use plonky2_maybe_rayon::*;
        let bytes = self.cpu_slots.len() * 8;
        if bytes == 0 {
            return ctx.storage("hybrid witness state", (e.state.len() * 8) as u64);
        }
        let staging = ctx.upload_staging(bytes as u64);
        e.program.runtime.run(|| {
            self.cpu_slots
                .par_chunks(16384)
                .enumerate()
                .for_each(|(block, slots)| {
                    let begin = block * 16384 * 8;
                    let mut view = staging
                        .buffer
                        .slice(begin as u64..(begin + slots.len() * 8) as u64)
                        .get_mapped_range_mut()
                        .expect("boundary mapping");
                    let (words, _) = view.slice(..).into_chunks::<8>();
                    for (word, &slot) in words.into_iter().zip(slots) {
                        word.write(e.state[slot as usize].load(Ordering::Relaxed).to_le_bytes());
                    }
                })
        });
        let packed = ctx.storage("CPU boundary values", bytes as u64);
        ctx.finish_upload(staging, &packed, bytes as u64);
        let slots = self.cpu_slots_device.get().unwrap_or_else(|| {
            let words = self
                .cpu_slots
                .chunks(2)
                .map(|x| x[0] as u64 | ((x.get(1).copied().unwrap_or(0) as u64) << 32))
                .collect::<Vec<_>>();
            let b = ctx.upload("CPU boundary slots", &words);
            self.cpu_slots_device.set_if_within_limit(ctx, b.clone());
            b
        });
        let state = ctx.storage("hybrid witness state", (e.state.len() * 8) as u64);
        let kernels = ctx.witness_tail.get_or_init(|| Kernels::new(ctx));
        let params = ctx.uniform("scatter range", &[self.cpu_slots.len() as u32, 0, 0, 0]);
        let bind = ctx.bind(&kernels.scatter_layout, &[&params, &packed, &slots, &state]);
        let mut encoder = ctx.encoder("scatter CPU boundary");
        {
            let mut pass = ctx.begin_pass(&mut encoder, "scatter CPU boundary", None);
            pass.set_pipeline(&kernels.scatter);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (self.cpu_slots.len() as u32).div_ceil(256);
            let width = groups.min(65535);
            pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
        }
        ctx.submit_async(encoder);
        state
    }
    pub(super) fn submit(
        &self,
        e: &Executor<'_>,
        ctx: &GpuContext,
        state: &DeviceBuffer,
        seed: Option<&[u8; 32]>,
    ) -> Result<DeviceBuffer> {
        let kernels = ctx.witness_tail.get_or_init(|| Kernels::new(ctx));
        let upload_u32 = |label: &str, cache: &DeviceCache, words: &[u32]| {
            cache.get().unwrap_or_else(|| {
                let words = words
                    .chunks(2)
                    .map(|c| c[0] as u64 | ((c.get(1).copied().unwrap_or(0) as u64) << 32))
                    .collect::<Vec<_>>();
                let b = ctx.upload(label, &words);
                cache.set_if_within_limit(ctx, b.clone());
                b
            })
        };
        let meta = upload_u32("tail descriptors", &self.metadata_device, &self.metadata);
        let targets = upload_u32("tail targets", &self.targets_device, &e.program.targets);
        let coefficients = self.coefficients_device.get().unwrap_or_else(|| {
            let b = ctx.upload("tail coefficients", &e.program.coefficients);
            self.coefficients_device.set_if_within_limit(ctx, b.clone());
            b
        });
        let random = if seed.is_some() && !self.shared_random {
            ctx.dummy.clone()
        } else {
            ctx.upload("tail random values", e.random)
        };
        let key_words = crate::gpu::random::seed_words(seed.unwrap_or(&[0u8; 32]), 0x5749544e);
        let key = ctx.uniform("tail random key", &key_words);
        let error = ctx.storage("tail errors", 16);
        let mut encoder = ctx.encoder("terminal witness regions");
        encoder.clear_buffer(&error, 0, None);
        let cache_params = crate::gpu::resident_witness::experiment().contains("params");
        let binds = self
            .runs
            .iter()
            .enumerate()
            .map(|(index, &(base, count, poseidon))| {
                let create = || {
                    DeviceBuffer::unpooled(ctx.uniform(
                        "tail range",
                        &[
                            base,
                            count,
                            u32::from(seed.is_some()),
                            if self.checks {
                                e.program.roots.len() as u32
                            } else {
                                0
                            },
                        ],
                    ))
                };
                let params = if cache_params {
                    let cache = &self.params_cache[index][usize::from(seed.is_some())];
                    cache.get().unwrap_or_else(|| {
                        let p = create();
                        cache.set_if_within_limit(ctx, p.clone());
                        p
                    })
                } else {
                    create()
                };
                (
                    ctx.bind(
                        &kernels.layout,
                        &[
                            &params,
                            &meta,
                            &targets,
                            &coefficients,
                            &random,
                            state,
                            &error,
                            &key,
                        ],
                    ),
                    count,
                    poseidon,
                    self.run_kinds[index],
                )
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "gpu-profiling")]
        let detailed = ctx.pass_timeline.is_some()
            && crate::util::switches::flag("PLONKY2_GPU_WITNESS_KERNEL_TIMELINE");
        #[cfg(not(feature = "gpu-profiling"))]
        let detailed = false;
        if detailed {
            for (bind, count, poseidon, kind) in &binds {
                let label = format!(
                    "witness {}",
                    match kind {
                        0 => "Poseidon",
                        2 => "Equality",
                        _ => "Other",
                    }
                );
                let mut pass = ctx.begin_pass(&mut encoder, &label, None);
                pass.set_pipeline(if *poseidon {
                    &kernels.poseidon
                } else {
                    &kernels.cheap
                });
                pass.set_bind_group(0, bind, &[]);
                let groups = count.div_ceil(128);
                let width = groups.min(65535);
                pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
            }
        } else {
            let mut pass = ctx.begin_pass(&mut encoder, "terminal witness regions", None);
            for (bind, count, poseidon, _) in &binds {
                pass.set_pipeline(if *poseidon {
                    &kernels.poseidon
                } else {
                    &kernels.cheap
                });
                pass.set_bind_group(0, bind, &[]);
                let groups = count.div_ceil(128);
                let width = groups.min(65535);
                pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
            }
        }
        if self.checks && e.program.alias_count > 0 {
            let params = ctx.uniform(
                "alias range",
                &[e.program.alias_pair_offset, e.program.alias_count, 0, 0],
            );
            let bind = ctx.bind(&kernels.alias_layout, &[&params, &targets, state, &error]);
            let mut pass = ctx.begin_pass(&mut encoder, "validate witness aliases", None);
            pass.set_pipeline(&kernels.alias);
            pass.set_bind_group(0, &bind, &[]);
            let groups = e.program.alias_count.div_ceil(256);
            let width = groups.min(65535);
            pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
        }
        ctx.submit_async(encoder);
        Ok(error)
    }
}

pub(crate) struct Kernels {
    layout: wgpu::BindGroupLayout,
    cheap: wgpu::ComputePipeline,
    poseidon: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    scatter: wgpu::ComputePipeline,
    alias_layout: wgpu::BindGroupLayout,
    alias: wgpu::ComputePipeline,
}
impl Kernels {
    fn new(ctx: &GpuContext) -> Self {
        let layout = crate::gpu::storage_layout(
            &ctx.device,
            "witness tail",
            &[
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, false),
                (false, false),
                (true, true),
            ],
        );
        let mut source = crate::gpu::shader::prelude_with_mds(ctx.flavor);
        source.push_str(crate::gpu::random::CHACHA);
        source.push_str(KERNEL);
        if crate::gpu::resident_witness::experiment().contains("dense") {
            source.push_str(&crate::gpu::shader::witness_dense_trace());
        } else {
            source.push_str(&trace());
        }
        let module = crate::gpu::module(&ctx.device, "witness tail", &source);
        let scatter_layout = crate::gpu::storage_layout(
            &ctx.device,
            "CPU boundary scatter",
            &[(true, true), (false, true), (false, true), (false, false)],
        );
        let scatter_module = crate::gpu::module(&ctx.device, "CPU boundary scatter", SCATTER);
        let scatter =
            crate::gpu::pipeline(&ctx.device, &scatter_layout, &scatter_module, "scatter");
        let alias_layout = crate::gpu::storage_layout(
            &ctx.device,
            "witness alias check",
            &[(true, true), (false, true), (false, true), (false, false)],
        );
        let alias_module = crate::gpu::module(&ctx.device, "witness alias check", ALIASES);
        let alias = crate::gpu::pipeline(&ctx.device, &alias_layout, &alias_module, "validate");
        Self {
            alias_layout,
            alias,
            scatter_layout,
            scatter,
            cheap: crate::gpu::pipeline(&ctx.device, &layout, &module, "cheap"),
            poseidon: crate::gpu::pipeline(&ctx.device, &layout, &module, "poseidon"),
            layout,
        }
    }
}
const KERNEL: &str = r#"
struct Params {base:u32,count:u32,pad0:u32,pad1:u32}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> metadata:array<u32>;
@group(0) @binding(2) var<storage,read> targets:array<u32>;
@group(0) @binding(3) var<storage,read> coefficients:array<Field>;
@group(0) @binding(4) var<storage,read> random:array<Field>;
@group(0) @binding(5) var<storage,read_write> state:array<Field>;
@group(0) @binding(6) var<storage,read_write> errors:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> random_key:RandomKey;
struct Op {kind:u32,offset:u32,count:u32,lane:u32,co:u32,inputs:u32,outputs:u32,random_on_device:u32}
fn operation(id:vec3<u32>,groups:vec3<u32>)->Op {
 let index=id.x+id.y*groups.x*128u;
 let i=(params.base+index)*8u;
 return Op(metadata[i],metadata[i+1u],metadata[i+2u],metadata[i+3u],metadata[i+4u],metadata[i+5u],metadata[i+6u],metadata[i+7u]);
}
fn inp(g:Op,pos:u32)->Field {return state[targets[g.offset+pos*g.count+g.lane]];}
fn coef(g:Op,pos:u32)->Field {return coefficients[g.co+pos*g.count+g.lane];}
fn emit(g:Op,pos:u32,value:Field) {
 let slot=targets[g.offset+(g.inputs+pos)*g.count+g.lane];let v=canonical(value);
 if(slot<params.pad1){if(any(state[slot]!=v)){atomicOr(&errors[0],2u);}}else{state[slot]=v;}
}
fn fail() {atomicOr(&errors[0],1u);}
fn chain_read(g:Op,v:Field,locals:array<Field,8>)->Field {
 if ((v.y & 0x80000000u)!=0u) {return locals[v.x];}
 return inp(g,v.x);
}
@compute @workgroup_size(128)
fn cheap(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let index=id.x+id.y*groups.x*128u;if(index>=params.count){return;}
 let g=operation(id,groups);
 switch g.kind {
 case 1u: {emit(g,0u,add_field(mul_field(mul_field(inp(g,0u),inp(g,1u)),coef(g,0u)),mul_field(inp(g,2u),coef(g,1u))));}
 case 2u: {emit(g,0u,inp(g,0u));}
 case 3u: {emit(g,0u,coef(g,0u));}
 case 4u: {let x=canonical(inp(g,0u));let y=canonical(inp(g,1u));if(all(x==y)){emit(g,0u,Field(1u,0u));emit(g,1u,Field(0u,0u));}else{emit(g,0u,Field(0u,0u));emit(g,1u,inv_field(sub_field(x,y)));}}
 case 5u: {let p=ext_mul(inp(g,0u),inp(g,1u),inp(g,2u),inp(g,3u));emit(g,0u,add_field(mul_field(p[0],coef(g,0u)),mul_field(inp(g,4u),coef(g,1u))));emit(g,1u,add_field(mul_field(p[1],coef(g,0u)),mul_field(inp(g,5u),coef(g,1u))));}
 case 6u: {let p=ext_mul(inp(g,0u),inp(g,1u),inp(g,2u),inp(g,3u));emit(g,0u,mul_field(p[0],coef(g,0u)));emit(g,1u,mul_field(p[1],coef(g,0u)));}
 case 7u: {let index=canonical(inp(g,0u));if(index.y!=0u || index.x>=g.co){fail();return;}emit(g,0u,inp(g,1u+index.x));for(var i=1u;i<g.outputs;i++){emit(g,i,Field((index.x>>(i-1u))&1u,0u));}}
 case 17u: {var value:Field;if(params.pad0!=0u && g.random_on_device!=0u){value=random_field(random_key,coef(g,0u).x);}else{value=random[coef(g,0u).x];}if(value.y==0xffffffffu && value.x>=1u){fail();return;}emit(g,0u,value);}
 case 18u: {let d=canonical(inp(g,0u));var v=Field(0u,0u);if(any(d!=Field(0u,0u))){v=mul_field(inp(g,1u),inv_field(d));}emit(g,0u,v);}
 case 26u: {var locals:array<Field,8>;for(var j=0u;j<g.outputs;j++){let a=chain_read(g,coef(g,5u*j),locals);let b=chain_read(g,coef(g,5u*j+1u),locals);let c=chain_read(g,coef(g,5u*j+2u),locals);locals[j]=add_field(mul_field(mul_field(a,b),coef(g,5u*j+3u)),mul_field(c,coef(g,5u*j+4u)));emit(g,j,locals[j]);}}
 case 27u: {var v=Field(0u,0u);for(var j=g.inputs;j>0u;j--){let b=canonical(inp(g,j-1u));if(b.y!=0u || b.x>1u){fail();return;}v=add_field(add_field(v,v),b);}emit(g,0u,v);}
 default: {fail();}
 }
}
@compute @workgroup_size(128)
fn poseidon(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let index=id.x+id.y*groups.x*128u;if(index>=params.count){return;}
 trace_poseidon(operation(id,groups));
}
"#;

fn trace() -> String {
    let mut s=String::from("fn trace_poseidon(g:Op) { var a:array<Field,12>; for(var i=0u;i<12u;i++){a[i]=inp(g,i);} let swap=canonical(inp(g,12u)); if(swap.y!=0u || swap.x>1u){fail();return;}\n");
    for i in 0..4 {
        writeln!(
            s,
            "emit(g,{i}u,mul_field(swap,sub_field(a[{}],a[{i}])));",
            i + 4
        )
        .unwrap();
    }
    s.push_str("if(swap.x==1u){for(var i=0u;i<4u;i++){let t=a[i];a[i]=a[i+4u];a[i+4u]=t;}}\n");
    let mut out = 4;
    let literal = crate::gpu::shader::literal;
    for r in 0..4 {
        for i in 0..12 {
            writeln!(
                s,
                "a[{i}]=add_field(a[{i}],{});",
                literal(ALL_ROUND_CONSTANTS[r * 12 + i])
            )
            .unwrap();
        }
        if r != 0 {
            for i in 0..12 {
                writeln!(s, "emit(g,{out}u,a[{i}]);").unwrap();
                out += 1;
            }
        }
        s.push_str("for(var i=0u;i<12u;i++){a[i]=sbox(a[i]);} a=mds12(a);\n");
    }
    for i in 0..12 {
        writeln!(
            s,
            "a[{i}]=add_field(a[{i}],{});",
            literal(G::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i])
        )
        .unwrap();
    }
    s.push_str("var b:array<Field,12>;b[0]=a[0];\n");
    for c in 1..12 {
        let mut expr = "Field(0u,0u)".to_owned();
        for r in 1..12 {
            expr = format!(
                "add_field({expr},mul_field(a[{r}],{}))",
                literal(G::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1])
            );
        }
        writeln!(s, "b[{c}]={expr};").unwrap();
    }
    s.push_str("a=b;\n");
    for r in 0..22 {
        writeln!(s, "emit(g,{out}u,a[0]);a[0]=sbox(a[0]);").unwrap();
        out += 1;
        if r != 21 {
            writeln!(
                s,
                "a[0]=add_field(a[0],{});",
                literal(G::FAST_PARTIAL_ROUND_CONSTANTS[r])
            )
            .unwrap();
        }
        let mut expr = format!(
            "mul_field(a[0],{})",
            literal(G::MDS_MATRIX_CIRC[0] + G::MDS_MATRIX_DIAG[0])
        );
        for i in 1..12 {
            expr = format!(
                "add_field({expr},mul_field(a[{i}],{}))",
                literal(G::FAST_PARTIAL_ROUND_W_HATS[r][i - 1])
            );
        }
        writeln!(s, "b[0]={expr};").unwrap();
        for i in 1..12 {
            writeln!(
                s,
                "b[{i}]=add_field(a[{i}],mul_field(a[0],{}));",
                literal(G::FAST_PARTIAL_ROUND_VS[r][i - 1])
            )
            .unwrap();
        }
        s.push_str("a=b;\n");
    }
    for r in 26..30 {
        for i in 0..12 {
            writeln!(
                s,
                "a[{i}]=add_field(a[{i}],{});emit(g,{out}u,a[{i}]);",
                literal(ALL_ROUND_CONSTANTS[r * 12 + i])
            )
            .unwrap();
            out += 1;
        }
        s.push_str("for(var i=0u;i<12u;i++){a[i]=sbox(a[i]);} a=mds12(a);\n");
    }
    for i in 0..12 {
        writeln!(s, "emit(g,{out}u,a[{i}]);").unwrap();
        out += 1;
    }
    assert_eq!(out, 122);
    s.push_str("}\n");
    s
}

const SCATTER: &str = r#"
@group(0) @binding(0) var<uniform> count:vec4<u32>;
@group(0) @binding(1) var<storage,read> packed:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> slots:array<u32>;
@group(0) @binding(3) var<storage,read_write> state:array<vec2<u32>>;
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let i=id.x+id.y*groups.x*256u;if(i<count.x){state[slots[i]]=packed[i];}
}
"#;

#[cfg(feature = "witness-benchmarks")]
fn analyze(p: &RegionProgram, public: &[u32]) {
    let supported = |k: Kind| {
        matches!(
            k,
            Kind::Poseidon
                | Kind::Arithmetic
                | Kind::Copy
                | Kind::Constant
                | Kind::Equality
                | Kind::ExtensionArithmetic
                | Kind::ExtensionMultiply
                | Kind::ArithmeticChain
                | Kind::RandomValue
                | Kind::ConditionalZero
                | Kind::BitSum
        )
    };
    let current = (0..Kind::COUNT)
        .filter(|&k| supported(Kind::from_code(k).unwrap()))
        .fold(0u128, |mask, k| mask | (1u128 << k));
    let fast = p.fast.get().unwrap();
    let estimate = |support: u128, checks: bool| {
        let mut cpu = vec![false; p.state_slots as usize];
        cpu[..p.roots.len()].fill(true);
        for &s in public {
            if s != u32::MAX {
                cpu[s as usize] = true;
            }
        }
        if !checks {
            for i in 0..p.alias_count as usize {
                cpu[p.targets[p.alias_pair_offset as usize + 2 * i] as usize] = true;
                cpu[p.targets[p.alias_pair_offset as usize + 2 * i + 1] as usize] = true;
            }
        }
        let mut counts = [0u64; Kind::COUNT];
        for groups in p.groups.iter().rev() {
            for g in groups {
                for k in 0..g.count {
                    let target = |pos: u32| {
                        p.targets[(g.targets_offset + pos * g.count + k) as usize] as usize
                    };
                    let offload = support & (1u128 << g.kind.code()) != 0
                        && (g.inputs..g.inputs + g.outputs).all(|pos| {
                            let s = target(pos);
                            if checks && s < p.roots.len() {
                                true
                            } else {
                                !cpu[s] && s >= p.roots.len() && fast.single[s] != 0
                            }
                        });
                    if offload {
                        counts[g.kind.code()] += 1;
                    } else {
                        for pos in 0..g.inputs {
                            cpu[target(pos)] = true;
                        }
                    }
                }
            }
        }
        counts
    };
    let mut totals = [0u64; Kind::COUNT];
    for g in p.groups.iter().flatten() {
        totals[g.kind.code()] += g.count as u64;
    }
    eprintln!("GPU_CUT_TOTALS {totals:?}");
    for (label, mask, checks) in [
        ("current", current, false),
        ("all-ops", u128::MAX, false),
        ("current-device-checks", current, true),
        ("all-device-checks", u128::MAX, true),
    ] {
        let counts = estimate(mask, checks);
        eprintln!("GPU_CUT_ESTIMATE label={label} counts={counts:?}");
    }
    for k in 0..Kind::COUNT {
        if current & (1u128 << k) == 0 {
            let counts = estimate(current | (1u128 << k), true);
            eprintln!(
                "GPU_CUT_EXTRA kind={:?} counts={counts:?}",
                Kind::from_code(k).unwrap()
            );
        }
    }
}

const ALIASES: &str = r#"
@group(0) @binding(0) var<uniform> range:vec4<u32>;
@group(0) @binding(1) var<storage,read> targets:array<u32>;
@group(0) @binding(2) var<storage,read> state:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> errors:array<atomic<u32>>;
@compute @workgroup_size(256)
fn validate(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let i=id.x+id.y*groups.x*256u;if(i>=range.y){return;}
 let offset=range.x+2u*i;
 if(any(state[targets[offset]]!=state[targets[offset+1u]])){atomicOr(&errors[0],4u);}
}
"#;

#[cfg(test)]
mod trace_tests {
    use super::*;
    use crate::field::types::{Field, Field64};
    #[test]
    fn dense_rounds_match_all_122_trace_values() {
        for case in 0..256u64 {
            let input = core::array::from_fn(|i| {
                G::from_canonical_u64(match case % 4 {
                    0 => 0,
                    1 => G::ORDER - 1,
                    2 => i as u64,
                    _ => case.wrapping_mul(0x123456789).wrapping_add(i as u64 * 1777),
                })
            });
            for swap in [G::ZERO, G::ONE] {
                let expected = crate::hash::poseidon_trace_simd::trace(input, swap);
                let mut state = input;
                let mut out = Vec::new();
                for i in 0..4 {
                    out.push(swap * (state[i + 4] - state[i]));
                }
                if swap == G::ONE {
                    for i in 0..4 {
                        state.swap(i, i + 4);
                    }
                }
                for round in 0..30 {
                    <G as Poseidon>::constant_layer_field(&mut state, round);
                    if (4..26).contains(&round) {
                        out.push(state[0]);
                        state[0] = <G as Poseidon>::sbox_monomial(state[0]);
                    } else {
                        if round != 0 {
                            out.extend(state);
                        }
                        <G as Poseidon>::sbox_layer(&mut state);
                    }
                    state = <G as Poseidon>::mds_layer(&state);
                }
                out.extend(state);
                assert_eq!(out.len(), 122);
                for (i, (a, b)) in out.iter().zip(expected).enumerate() {
                    assert_eq!(*a, b, "case={case} swap={swap:?} word={i}");
                }
            }
        }
    }
}
