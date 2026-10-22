//! Merkle trees on the device: the leaves of a level are hashed in one
//! dispatch per level, straight into Plonky2's flattened digest layout.

use plonky2_maybe_rayon::*;

use crate::gpu::shader::HASH_WORKGROUP;
use crate::gpu::{context, field_supported, field_words, DeviceBuffer, GpuContext};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleLeaves;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// Where a leaf's elements come from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LeafSource {
    /// `src[i * leaf_len + e]`.
    Rows { leaf_len: u32 },
    /// `columns` columns of `2^log_n` elements; leaf `i` has
    /// `columns << arity_bits` elements, element `e` being row
    /// `rev(i * 2^arity_bits + e / columns)` of column `e % columns`.
    Columns {
        columns: u32,
        log_n: u32,
        arity_bits: u32,
    },
}

/// A tree's digests and cap on the device. The leaves are hashed by one
/// submission and the levels by a second, so that a host that only needs
/// the leaves (or the rows the leaf kernel wrote) can go on while the
/// levels are still being hashed.
#[derive(Debug)]
pub(crate) struct DeviceTree {
    pub digests: DeviceBuffer,
    pub cap: DeviceBuffer,
    pub num_digests: usize,
    pub cap_len: usize,
    /// The submission hashing the leaves.
    pub leaves_done: wgpu::SubmissionIndex,
    /// The submission hashing the levels above them.
    pub done: wgpu::SubmissionIndex,
    timestamps: Option<super::timestamps::MerkleTimestamps>,
}

/// Hashes `2^log_leaves` leaves of `source` into a tree with a cap of height
/// `cap_height`. With `rows`, every gathered leaf is also written there, row
/// by row, as the leaf hashing reads it.
pub(crate) fn build_tree(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    source: LeafSource,
    log_leaves: u32,
    cap_height: u32,
    rows: Option<&wgpu::Buffer>,
) -> DeviceTree {
    assert!(cap_height <= log_leaves);
    let num_leaves = 1usize << log_leaves;
    let num_layers = log_leaves - cap_height;
    let cap_len = 1usize << cap_height;
    let num_digests = 2 * (num_leaves - cap_len);
    let tree_len = num_digests >> cap_height;
    let digests = ctx.storage("digests", num_digests as u64 * 32);
    let cap = ctx.storage("cap", cap_len as u64 * 32);
    let (leaf_len, layout, columns, log_n, arity_bits) = match source {
        LeafSource::Rows { leaf_len } => (leaf_len, 0, 0, 0, 0),
        LeafSource::Columns {
            columns,
            log_n,
            arity_bits,
        } => (columns << arity_bits, 1, columns, log_n, arity_bits),
    };
    let params = |count: u32, layer: u32| {
        ctx.uniform(
            "hash params",
            &[
                count,
                leaf_len,
                layout,
                columns,
                log_n,
                arity_bits,
                log_leaves,
                layer,
                num_layers,
                tree_len as u32,
                rows.is_some() as u32,
                0,
            ],
        )
    };
    let rows = rows.unwrap_or(&ctx.dummy_rw);
    let mut bindings = Vec::with_capacity(num_layers as usize + 1);
    bindings.push(ctx.bind_hash(&params(num_leaves as u32, 0), src, &digests, &cap, rows));
    // The levels do not bind the rows, so that the host can map them as soon
    // as the leaves are hashed.
    for layer in 1..=num_layers {
        bindings.push(ctx.bind_hash(
            &params((num_leaves >> layer) as u32, layer),
            src,
            &digests,
            &cap,
            &ctx.dummy_rw,
        ));
    }
    let mut encoder = ctx.encoder("merkle leaves");
    // Named by shape so captures and traces can tell the commitments apart.
    let leaves_label = format!("merkle leaves {num_leaves} x {leaf_len}");
    let timestamps = super::timestamps::MerkleTimestamps::new(ctx, num_layers > 0);
    {
        let mut pass = ctx.begin_pass(
            &mut encoder,
            &leaves_label,
            timestamps.as_ref().map(|t| t.writes(0)),
        );
        pass.set_pipeline(&ctx.hash_leaves);
        pass.set_bind_group(0, &bindings[0], &[]);
        pass.dispatch_workgroups((num_leaves as u32).div_ceil(HASH_WORKGROUP), 1, 1);
    }
    let leaves_done = ctx.submit_async(encoder);
    let mut encoder = ctx.encoder("merkle levels");
    {
        let mut pass = ctx.begin_pass(
            &mut encoder,
            "merkle levels",
            timestamps.as_ref().map(|t| t.writes(2)),
        );
        pass.set_pipeline(&ctx.hash_nodes);
        for layer in 1..=num_layers {
            pass.set_bind_group(0, &bindings[layer as usize], &[]);
            pass.dispatch_workgroups(
                ((num_leaves >> layer) as u32).div_ceil(HASH_WORKGROUP),
                1,
                1,
            );
        }
    }
    let done = ctx.submit_async(encoder);
    DeviceTree {
        digests,
        cap,
        num_digests,
        cap_len,
        leaves_done,
        done,
        timestamps,
    }
}

fn read_hashes<F: RichField, H: Hasher<F>>(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    count: usize,
    done: &wgpu::SubmissionIndex,
    timing: Option<&mut crate::util::timing::TimingTree>,
) -> Vec<H::Hash> {
    ctx.read_after_with_timing(buffer, count as u64 * 32, Some(done), timing, |bytes| {
        bytes
            .par_chunks_exact(32)
            .map(H::Hash::from_bytes)
            .collect()
    })
}

impl DeviceTree {
    /// Reads the digests and the cap back as hashes.
    pub(crate) fn read<F: RichField, H: Hasher<F>>(
        &self,
        ctx: &GpuContext,
    ) -> DigestsAndCap<H::Hash> {
        self.read_with_timing::<F, H>(ctx, None)
    }

    pub(crate) fn read_with_timing<F: RichField, H: Hasher<F>>(
        &self,
        ctx: &GpuContext,
        mut timing: Option<&mut crate::util::timing::TimingTree>,
    ) -> DigestsAndCap<H::Hash> {
        let result = (
            read_hashes::<F, H>(
                ctx,
                &self.digests,
                self.num_digests,
                &self.done,
                timing.as_deref_mut(),
            ),
            read_hashes::<F, H>(ctx, &self.cap, self.cap_len, &self.done, timing),
        );
        self.gpu_timings(ctx);
        result
    }

    pub(crate) fn gpu_timings(
        &self,
        ctx: &GpuContext,
    ) -> Option<super::timestamps::MerkleGpuTimings> {
        self.timestamps.as_ref().map(|t| t.read(ctx, &self.done))
    }

    /// Prototype: copy only authentication siblings from the resident digest
    /// array into one compact readback. Duplicate queries retain their order.
    /// Kept outside the default prover path while public host digest storage
    /// and serialization continue to use the existing full readback.
    #[cfg(all(test, feature = "gpu-profiling"))]
    pub(crate) fn read_query_paths<F: RichField, H: Hasher<F>>(
        &self,
        ctx: &GpuContext,
        leaves_len: usize,
        cap_height: usize,
        indices: &[usize],
    ) -> Vec<Vec<H::Hash>> {
        let log_leaves = log2_strict(leaves_len);
        assert!(cap_height <= log_leaves);
        let depth = log_leaves - cap_height;
        assert_eq!(self.cap_len, 1usize << cap_height);
        assert_eq!(self.num_digests, 2 * (leaves_len - self.cap_len));
        assert!(indices.iter().all(|&i| i < leaves_len));
        if depth == 0 || indices.is_empty() {
            return vec![Vec::new(); indices.len()];
        }
        let compact = ctx.storage("merkle query siblings", (indices.len() * depth * 32) as u64);
        let mut encoder = ctx.encoder("gather merkle query siblings");
        let tree_len = self.num_digests >> cap_height;
        for (query, &index) in indices.iter().enumerate() {
            let tree_start = (index >> depth) * tree_len;
            let mut pair = index & ((1 << depth) - 1);
            for level in 0..depth {
                let parity = pair & 1;
                pair >>= 1;
                let sibling =
                    tree_start + 2 * ((pair << (level + 1)) + (1 << level) - 1) + 1 - parity;
                encoder.copy_buffer_to_buffer(
                    &self.digests,
                    sibling as u64 * 32,
                    &compact,
                    ((query * depth + level) * 32) as u64,
                    32,
                );
            }
        }
        let done = ctx.submit_async(encoder);
        let hashes = read_hashes::<F, H>(ctx, &compact, indices.len() * depth, &done, None);
        hashes
            .chunks_exact(depth)
            .map(<[H::Hash]>::to_vec)
            .collect()
    }

    /// Reads only the cap, leaving the digests on the device.
    pub(crate) fn read_cap<F: RichField, H: Hasher<F>>(
        &self,
        ctx: &GpuContext,
        timing: Option<&mut crate::util::timing::TimingTree>,
    ) -> Vec<H::Hash> {
        let cap = read_hashes::<F, H>(ctx, &self.cap, self.cap_len, &self.done, timing);
        self.gpu_timings(ctx);
        cap
    }
}

/// Whether the GPU can build trees for this field and hasher.
pub(crate) fn supported<F: RichField, H: Hasher<F>>() -> bool {
    H::GPU_POSEIDON && field_supported::<F>()
}

/// The digests and the cap of a tree.
pub(crate) type DigestsAndCap<H> = (Vec<H>, Vec<H>);

/// The digests and cap of the tree over `leaves`, if the GPU takes it.
pub(crate) fn try_new<F: RichField, H: Hasher<F>>(
    leaves: &MerkleLeaves<F>,
    cap_height: usize,
) -> Option<DigestsAndCap<H::Hash>> {
    if !supported::<F, H>() {
        return None;
    }
    let ctx = context()?;
    let ctx = &*ctx;
    let log_leaves = log2_strict(leaves.len()) as u32;
    if log_leaves < crate::gpu::MIN_LOG_LEAVES
        || !ctx.fits(leaves.num_values() as u64 * 8)
        || (leaves.len() as u32).div_ceil(HASH_WORKGROUP) > ctx.max_workgroups
    {
        return None;
    }
    Some(build_from_rows::<F, H>(
        ctx,
        leaves,
        log_leaves,
        cap_height as u32,
    ))
}

/// Builds the tree over `leaves`.
pub(crate) fn build_from_rows<F: RichField, H: Hasher<F>>(
    ctx: &GpuContext,
    leaves: &MerkleLeaves<F>,
    log_leaves: u32,
    cap_height: u32,
) -> DigestsAndCap<H::Hash> {
    let leaf_len = leaves.leaf_len;
    let src = ctx.upload("leaves", &field_words(leaves.as_flat_slice()));
    let tree = build_tree(
        ctx,
        &src,
        LeafSource::Rows {
            leaf_len: leaf_len as u32,
        },
        log_leaves,
        cap_height,
        None,
    );
    tree.read::<F, H>(ctx)
}
