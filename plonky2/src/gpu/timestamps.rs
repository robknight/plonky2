//! Opt-in device timings, distinct from CPU submission/readback wall times.

use std::sync::OnceLock;

use super::GpuContext;

/// Device span across one or more quotient compute passes. Diagnostic only.
#[cfg(feature = "gpu-profiling")]
pub(crate) struct ComputeSpan {
    query: wgpu::QuerySet,
    resolved: wgpu::Buffer,
}

#[cfg(feature = "gpu-profiling")]
impl ComputeSpan {
    pub fn new(ctx: &GpuContext) -> Option<Self> {
        if !ctx
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            || !crate::util::switches::flag("PLONKY2_GPU_TIMESTAMPS")
        {
            return None;
        }
        let query = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("quotient timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let mut usage = wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC;
        if ctx.mappable {
            usage |= wgpu::BufferUsages::MAP_READ;
        }
        let resolved = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quotient resolved timestamps"),
            size: 16,
            usage,
            mapped_at_creation: false,
        });
        Some(Self { query, resolved })
    }

    pub fn writes(&self, first: bool, last: bool) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query,
            beginning_of_pass_write_index: first.then_some(0),
            end_of_pass_write_index: last.then_some(1),
        }
    }

    pub fn read(&self, ctx: &GpuContext, done: &wgpu::SubmissionIndex) -> f64 {
        ctx.wait(done);
        let mut encoder = ctx.encoder("resolve quotient timestamps");
        encoder.resolve_query_set(&self.query, 0..2, &self.resolved, 0);
        let ready = ctx.submit_async(encoder);
        let t = ctx.read_after(&self.resolved, 16, Some(&ready), |bytes| {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect::<Vec<_>>()
        });
        if t[0] == 0 || t[1] < t[0] {
            return f64::NAN;
        }
        (t[1] - t[0]) as f64 * ctx.queue.get_timestamp_period() as f64 / 1_000_000.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MerkleGpuTimings {
    pub leaves_ms: f64,
    pub levels_ms: f64,
}

#[derive(Debug)]
pub(crate) struct MerkleTimestamps {
    query: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    sample: OnceLock<MerkleGpuTimings>,
    has_levels: bool,
}

impl MerkleTimestamps {
    pub fn new(ctx: &GpuContext, has_levels: bool) -> Option<Self> {
        if !cfg!(feature = "gpu-profiling")
            || !ctx
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            || !crate::util::switches::flag("PLONKY2_GPU_TIMESTAMPS")
        {
            return None;
        }
        let query = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("merkle timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        });
        let mut usage = wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC;
        if ctx.mappable {
            usage |= wgpu::BufferUsages::MAP_READ;
        }
        let resolved = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merkle resolved timestamps"),
            size: 32,
            usage,
            mapped_at_creation: false,
        });
        Some(Self {
            query,
            resolved,
            sample: OnceLock::new(),
            has_levels,
        })
    }

    pub fn writes(&self, first: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query, 0..4, &self.resolved, 0);
    }

    pub fn read(&self, ctx: &GpuContext, done: &wgpu::SubmissionIndex) -> MerkleGpuTimings {
        *self.sample.get_or_init(|| {
            // Metal pass-boundary counters may become visible only after the
            // measured command buffer completes. Resolve in a later submission,
            // rather than reading zero samples from that same command buffer.
            ctx.wait(done);
            let mut encoder = ctx.encoder("resolve merkle timestamps");
            self.resolve(&mut encoder);
            let resolved = ctx.submit_async(encoder);
            let stamps = ctx.read_after(&self.resolved, 32, Some(&resolved), |bytes| {
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect::<Vec<_>>()
            });
            let to_ms = ctx.queue.get_timestamp_period() as f64 / 1_000_000.0;
            log::debug!(
                "GPU_MERKLE_TIMESTAMP_RAW stamps={stamps:?} period_ns={}",
                ctx.queue.get_timestamp_period()
            );
            let duration = |start: u64, end: u64| {
                if start == 0 || end < start {
                    log::warn!("GPU Merkle timestamp unavailable: start={start} end={end}");
                    f64::NAN
                } else {
                    (end - start) as f64 * to_ms
                }
            };
            let result = MerkleGpuTimings {
                leaves_ms: duration(stamps[0], stamps[1]),
                levels_ms: if self.has_levels {
                    duration(stamps[2], stamps[3])
                } else {
                    0.0
                },
            };
            log::debug!(
                "GPU_MERKLE_TIMING leaves_ms={:.6} levels_ms={:.6}",
                result.leaves_ms,
                result.levels_ms
            );
            result
        })
    }
}

/// Whole-proof device timeline (`PLONKY2_GPU_PASS_TIMELINE=1`, profiling
/// builds): every compute pass begun through `GpuContext::begin_pass` without
/// its own timestamps gets a begin and end timestamp query, and `flush`
/// resolves them all after the proof and logs one `GPU_PASS` line per pass in
/// device nanoseconds. Copies and queue writes are not passes and show as
/// idle time between them.
#[cfg(feature = "gpu-profiling")]
pub(crate) struct PassTimeline {
    chunks: std::sync::Mutex<Vec<(wgpu::QuerySet, Vec<String>)>>,
    flushes: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu-profiling")]
const TIMELINE_CHUNK: usize = 512;

#[cfg(feature = "gpu-profiling")]
impl PassTimeline {
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            || !crate::util::switches::flag("PLONKY2_GPU_PASS_TIMELINE")
        {
            return None;
        }
        log::warn!("plonky2 gpu: recording a device timestamp pair for every compute pass");
        Some(Self {
            chunks: std::sync::Mutex::new(Vec::new()),
            flushes: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// The query set and first index of a new pass's begin/end pair.
    pub fn slot(&self, device: &wgpu::Device, label: &str) -> (wgpu::QuerySet, u32) {
        let mut chunks = self.chunks.lock().unwrap();
        if chunks
            .last()
            .is_none_or(|(_, labels)| labels.len() == TIMELINE_CHUNK)
        {
            let query = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timeline"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * TIMELINE_CHUNK as u32,
            });
            chunks.push((query, Vec::new()));
        }
        let (query, labels) = chunks.last_mut().unwrap();
        labels.push(label.to_string());
        (query.clone(), 2 * (labels.len() as u32 - 1))
    }

    /// Waits for the device, resolves and logs every recorded pass, and
    /// starts a new record. Called at the end of each proof.
    pub fn flush(&self, ctx: &GpuContext) {
        let chunks = std::mem::take(&mut *self.chunks.lock().unwrap());
        if chunks.is_empty() {
            return;
        }
        let flush = self
            .flushes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("waiting for the GPU");
        let period = ctx.queue.get_timestamp_period() as f64;
        for (query, labels) in chunks {
            let count = 2 * labels.len();
            let mut usage = wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC;
            if ctx.mappable {
                usage |= wgpu::BufferUsages::MAP_READ;
            }
            let resolved = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass timeline resolved"),
                size: (count * 8) as u64,
                usage,
                mapped_at_creation: false,
            });
            let mut encoder = ctx.encoder("resolve pass timeline");
            encoder.resolve_query_set(&query, 0..count as u32, &resolved, 0);
            let done = ctx.submit_async(encoder);
            let stamps = ctx.read_after(&resolved, (count * 8) as u64, Some(&done), |bytes| {
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect::<Vec<_>>()
            });
            for (k, label) in labels.iter().enumerate() {
                let (begin, end) = (stamps[2 * k], stamps[2 * k + 1]);
                log::info!(
                    "GPU_PASS flush={flush} label=\"{label}\" begin_ns={:.0} end_ns={:.0}",
                    begin as f64 * period,
                    end as f64 * period
                );
            }
        }
    }
}
