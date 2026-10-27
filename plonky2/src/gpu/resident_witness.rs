//! Wire values the compiled witness executor (`plonk::witness_program`)
//! uploaded through a staging buffer, handed to the prover in place of a host
//! witness matrix.
use std::sync::Arc;

use crate::gpu::{DeviceBuffer, GpuContext};

/// Wire values already on the device in column-major order, handed to the
/// prover in place of a host witness matrix.
pub(crate) struct ResidentWitness {
    pub ctx: Arc<GpuContext>,
    pub wire_values: DeviceBuffer,
    /// Fixed column-major cell -> compact state slot, or dense values when absent.
    pub wire_slots: Option<DeviceBuffer>,
    pub error: Option<DeviceBuffer>,
}

/// Benchmark-only pipeline experiments, selected before a process starts proving.
pub(crate) fn experiment() -> String {
    #[cfg(feature = "witness-benchmarks")]
    {
        std::env::var("PLONKY2_GPU_WITNESS_EXPERIMENT").unwrap_or_default()
    }
    #[cfg(not(feature = "witness-benchmarks"))]
    {
        String::new()
    }
}

pub(crate) struct CompactFft {
    pub layout: wgpu::BindGroupLayout,
    pub ordinary: wgpu::ComputePipeline,
    pub pipelined: wgpu::ComputePipeline,
}
impl GpuContext {
    pub(crate) fn compact_fft(&self) -> &CompactFft {
        self.compact_fft.get_or_init(|| {
            let layout = super::storage_layout(&self.device, "compact fft", &[
                (true,true),(false,true),(false,false),(false,false),
                (false,true),(false,true),(false,true),
            ]);
            let source = super::shader::fft_module(self.flavor)
                .replace("v = input[col * fp.in_stride + x];", "let slot = wire_slots[col * fp.in_stride + x]; if (slot != 0xffffffffu) { v = input[slot]; }");
            let source = format!("{source}\n@group(0) @binding(6) var<storage, read> wire_slots: array<u32>;\n");
            let module = super::module(&self.device, "compact fft", &source);
            CompactFft {
                ordinary: super::pipeline(&self.device, &layout, &module, "fft_group"),
                pipelined: super::pipeline(&self.device, &layout, &module, "fft_group_pipelined"),
                layout,
            }
        })
    }
}

impl ResidentWitness {
    pub(crate) fn read_columns<F: crate::field::types::PrimeField64>(
        &self,
        columns: usize,
        degree: usize,
    ) -> Vec<Vec<F>> {
        if let Some(slots) = &self.wire_slots {
            let state = self
                .ctx
                .read_u64(&self.wire_values, self.wire_values.size() as usize / 8);
            let slots = self.ctx.read_u64(slots, (columns * degree).div_ceil(2));
            (0..columns)
                .map(|column| {
                    (0..degree)
                        .map(|row| {
                            let i = column * degree + row;
                            let slot = (slots[i / 2] >> (32 * (i % 2))) as u32;
                            F::from_canonical_u64(if slot == u32::MAX {
                                0
                            } else {
                                state[slot as usize]
                            })
                        })
                        .collect()
                })
                .collect()
        } else {
            super::lde::read_columns::<F>(&self.ctx, &self.wire_values, columns, degree)
        }
    }
}

pub(crate) struct CompactPermutation {
    pub layout: wgpu::BindGroupLayout,
    pub rows: std::sync::Mutex<std::collections::HashMap<u32, Arc<wgpu::ComputePipeline>>>,
    pub rest: [wgpu::ComputePipeline; 3],
}
impl GpuContext {
    pub(crate) fn compact_permutation(&self) -> &CompactPermutation {
        self.compact_permutation.get_or_init(|| {
            let layout = super::storage_layout(
                &self.device,
                "compact permutation",
                &[
                    (true, true),
                    (false, true),
                    (false, true),
                    (false, true),
                    (false, true),
                    (false, false),
                    (false, false),
                    (false, true),
                ],
            );
            let source = super::shader::permutation_module(self.flavor);
            let module = super::module(&self.device, "compact permutation rest", &source);
            let rest = ["block_products", "block_prefix", "fill_columns"]
                .map(|entry| super::pipeline(&self.device, &layout, &module, entry));
            CompactPermutation {
                layout,
                rows: Default::default(),
                rest,
            }
        })
    }
    pub(crate) fn compact_permutation_rows(
        &self,
        chunks: u32,
        max_degree: usize,
    ) -> Arc<wgpu::ComputePipeline> {
        let c = self.compact_permutation();
        let key = if max_degree <= 2 { 0 } else { chunks };
        let mut cache = c.rows.lock().unwrap();
        cache.entry(key).or_insert_with(||{
            let source=if key==0 {super::shader::permutation_module_individual(self.flavor)} else {super::shader::permutation_module_with_chunks(self.flavor,chunks)};
            let source=source.replace("wires[j * pp.degree + i]","compact_wire(j * pp.degree + i)")
                .replace("wires[k * pp.degree + i]","compact_wire(k * pp.degree + i)");
            let source=format!("{source}\n@group(0) @binding(7) var<storage,read> wire_slots: array<u32>;\nfn compact_wire(i:u32)->Field {{ let slot=wire_slots[i]; if(slot==0xffffffffu){{return Field(0u,0u);}} return wires[slot]; }}\n");
            let module=super::module(&self.device,"compact row products",&source);
            Arc::new(super::pipeline(&self.device,&c.layout,&module,"row_products"))
        }).clone()
    }
}

impl ResidentWitness {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if let Some(error) = &self.error {
            anyhow::ensure!(
                self.ctx.read_u64(error, 2).iter().all(|x| *x == 0),
                "GPU witness rejected invalid operation inputs"
            );
        }
        Ok(())
    }
}
