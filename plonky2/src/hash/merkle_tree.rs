#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }
}

/// Internal storage: host access is explicit through `MerkleLeaves` methods.
#[derive(Clone, Debug)]
struct LeafValues<F: Copy> {
    host: Vec<F>,
    #[cfg(feature = "gpu")]
    device: Option<std::sync::Arc<DeviceLeaves<F>>>,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct ResidentBuffer {
    buffer: crate::gpu::DeviceBuffer,
    context: std::sync::Arc<crate::gpu::GpuContext>,
    done: wgpu::SubmissionIndex,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct DeviceLeaves<F: Copy> {
    // The full download and sparse readbacks share this lock. Once a full
    // host copy exists, take/drop the buffer AND context for all leaf clones.
    resident: std::sync::Mutex<Option<ResidentBuffer>>,
    count: usize,
    decode: fn(&[u8]) -> Vec<F>,
    full: std::sync::OnceLock<Vec<F>>,
    queries: std::sync::OnceLock<std::collections::HashMap<usize, Vec<F>>>,
    rows: std::sync::OnceLock<PreparedRows<F>>,
}

/// Rows copied out of a mapped device buffer for a CPU computation.
#[cfg(feature = "gpu")]
#[derive(Debug)]
struct PreparedRows<F> {
    /// Position of each leaf's row in `data`, `u32::MAX` when not prepared.
    index: Vec<u32>,
    data: Vec<F>,
}

#[cfg(feature = "gpu")]
impl<F: Copy> DeviceLeaves<F> {
    fn materialize(&self) -> &Vec<F> {
        self.full.get_or_init(|| {
            let mut resident = self.resident.lock().unwrap();
            let d = resident
                .as_ref()
                .expect("resident buffer before materialization");
            log::debug!(
                "gpu: downloading all {} leaf values to the host",
                self.count
            );
            let values = d.context.read_after(
                &d.buffer,
                (self.count * 8) as u64,
                Some(&d.done),
                self.decode,
            );
            // All GPU readbacks are complete. Host data now owns the contents.
            resident.take();
            values
        })
    }
}

impl<F: Copy> From<Vec<F>> for LeafValues<F> {
    fn from(host: Vec<F>) -> Self {
        Self {
            host,
            #[cfg(feature = "gpu")]
            device: None,
        }
    }
}

impl<F: Copy> LeafValues<F> {
    fn as_slice(&self) -> &[F] {
        #[cfg(feature = "gpu")]
        if let Some(d) = &self.device {
            return d.materialize();
        }
        &self.host
    }

    fn into_vec(self) -> Vec<F> {
        #[cfg(feature = "gpu")]
        if let Some(d) = self.device {
            d.materialize();
            return match std::sync::Arc::try_unwrap(d) {
                Ok(d) => d.full.into_inner().expect("materialized leaves"),
                Err(d) => d.materialize().clone(),
            };
        }
        self.host
    }
}

impl<F: Copy + PartialEq> PartialEq for LeafValues<F> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<F: Copy + Eq> Eq for LeafValues<F> {}
impl<F: Copy + Serialize> Serialize for LeafValues<F> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(serializer)
    }
}
impl<'de, F: Copy + Deserialize<'de>> Deserialize<'de> for LeafValues<F> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Vec::<F>::deserialize(deserializer)?.into())
    }
}

#[cfg(feature = "gpu")]
fn decode_leaf_fields<F: crate::field::types::PrimeField64>(bytes: &[u8]) -> Vec<F> {
    let words = bytes.as_chunks::<8>().0;
    let decode = |word: &[u8; 8]| F::from_noncanonical_u64(u64::from_le_bytes(*word));
    #[cfg(feature = "parallel")]
    if words.len() >= 1 << 17 {
        // Do not schedule Rayon work while initializing the shared OnceLock:
        // a stolen row-access task could wait on this same initialization.
        // Scoped workers perform only decoding, so concurrent readers cannot
        // deadlock the initializer. Small/sparse reads stay on this thread.
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8);
        let chunk_len = words.len().div_ceil(workers);
        let mut values = vec![F::ZERO; words.len()];
        std::thread::scope(|scope| {
            for (src, dst) in words.chunks(chunk_len).zip(values.chunks_mut(chunk_len)) {
                scope.spawn(move || {
                    for (word, value) in src.iter().zip(dst) {
                        *value = decode(word);
                    }
                });
            }
        });
        return values;
    }
    words.iter().map(decode).collect()
}

#[cfg(feature = "gpu")]
impl<F: crate::field::types::PrimeField64> MerkleLeaves<F> {
    pub(crate) fn from_device(
        context: std::sync::Arc<crate::gpu::GpuContext>,
        buffer: crate::gpu::DeviceBuffer,
        leaf_len: usize,
        num_leaves: usize,
        done: wgpu::SubmissionIndex,
    ) -> Self {
        let count = leaf_len * num_leaves;
        assert!(buffer.size() >= (count * 8) as u64);
        Self {
            values: LeafValues {
                host: Vec::new(),
                device: Some(std::sync::Arc::new(DeviceLeaves {
                    resident: std::sync::Mutex::new(Some(ResidentBuffer {
                        context,
                        buffer,
                        done,
                    })),
                    count,
                    decode: decode_leaf_fields::<F>,
                    full: Default::default(),
                    queries: Default::default(),
                    rows: Default::default(),
                })),
            },
            leaf_len,
            num_leaves,
            row_view: RowView::default(),
        }
    }
}

#[cfg(feature = "gpu")]
impl<F: Copy> MerkleLeaves<F> {
    /// Gather all queried rows in one submission/readback. A full CPU request
    /// still works. A later, different query set falls back to full host data.
    pub(crate) fn prepare_queries(&self, indices: &[usize]) {
        let Some(d) = &self.values.device else { return };
        if d.full.get().is_some() || indices.is_empty() {
            return;
        }
        d.queries.get_or_init(|| {
            let mut indices = indices.to_vec();
            indices.sort_unstable();
            indices.dedup();
            assert!(indices.iter().all(|&i| i < self.num_leaves));
            let resident = d.resident.lock().unwrap();
            // Materialization may have won the lock. get() will use host data.
            let Some(buffer) = resident.as_ref() else {
                return Default::default();
            };
            let ctx = &buffer.context;
            let row_bytes = (self.leaf_len * 8) as u64;
            let compact = ctx.storage("queried leaf rows", row_bytes * indices.len() as u64);
            let mut encoder = ctx.encoder("gather queried leaf rows");
            for (j, &index) in indices.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    &buffer.buffer,
                    index as u64 * row_bytes,
                    &compact,
                    j as u64 * row_bytes,
                    row_bytes,
                );
            }
            let done = ctx.submit_async(encoder);
            ctx.read_after(
                &compact,
                row_bytes * indices.len() as u64,
                Some(&done),
                |bytes| {
                    indices
                        .iter()
                        .enumerate()
                        .map(|(j, &index)| {
                            let start = j * row_bytes as usize;
                            (index, (d.decode)(&bytes[start..start + row_bytes as usize]))
                        })
                        .collect()
                },
            )
        });
    }

    /// Copies the rows a CPU computation will read into a compact host table
    /// that `get` consults before the full download. Only on devices whose
    /// buffers the host maps directly (unified memory), where the copy reads
    /// the GPU's own buffer; elsewhere, or once the full host copy exists,
    /// this does nothing and rows come from the full download. One set per
    /// leaves value: a later call is ignored. Runs on the calling thread and,
    /// for decoding, on scoped threads, never on the Rayon pool, because a
    /// stolen row-access task could wait on the lock held here.
    pub(crate) fn prepare_rows(&self, indices: &[usize])
    where
        F: Send,
    {
        let Some(d) = &self.values.device else { return };
        if d.full.get().is_some() || d.rows.get().is_some() || indices.is_empty() {
            return;
        }
        let resident = d.resident.lock().unwrap();
        // Another caller may have prepared rows while this one waited.
        if d.rows.get().is_some() {
            return;
        }
        let Some(buffer) = resident.as_ref() else {
            return;
        };
        if !buffer.context.mappable {
            return;
        }
        let started = std::time::Instant::now();
        let mut indices = indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        assert!(indices.iter().all(|&i| i < self.num_leaves));
        let row_bytes = self.leaf_len * 8;
        let data = buffer.context.read_after(
            &buffer.buffer,
            (d.count * 8) as u64,
            Some(&buffer.done),
            |bytes| {
                // Each scoped worker copies and decodes its own run of rows in
                // small blocks, so every decode call stays single-threaded.
                let workers = std::thread::available_parallelism()
                    .map_or(1, usize::from)
                    .clamp(1, 8);
                let chunk = indices.len().div_ceil(workers).max(1);
                let decode = d.decode;
                let parts = std::thread::scope(|scope| {
                    let handles = indices
                        .chunks(chunk)
                        .map(|run| {
                            scope.spawn(move || {
                                let mut out = Vec::with_capacity(run.len() * row_bytes / 8);
                                let mut block = Vec::with_capacity(256 * row_bytes);
                                for rows in run.chunks(256) {
                                    block.clear();
                                    for &i in rows {
                                        block.extend_from_slice(
                                            &bytes[i * row_bytes..(i + 1) * row_bytes],
                                        );
                                    }
                                    out.extend(decode(&block));
                                }
                                out
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().expect("leaf row worker panicked"))
                        .collect::<Vec<_>>()
                });
                // Consume the parts so each is freed once copied.
                let mut data = Vec::with_capacity(indices.len() * self.leaf_len);
                for part in parts {
                    data.extend(part);
                }
                data
            },
        );
        drop(resident);
        let mut index = vec![u32::MAX; self.num_leaves];
        for (k, &i) in indices.iter().enumerate() {
            index[i] = u32::try_from(k).expect("fewer than 2^32 prepared rows");
        }
        log::debug!(
            "gpu: prepared {} of {} leaf rows of {} values in {:?}",
            indices.len(),
            self.num_leaves,
            self.leaf_len,
            started.elapsed()
        );
        let _ = d.rows.set(PreparedRows { index, data });
    }

    /// True while a device copy is retained. Does not download any data.
    pub(crate) fn is_device_resident(&self) -> bool {
        self.values
            .device
            .as_ref()
            .is_some_and(|d| d.resident.lock().unwrap().is_some())
    }
}

/// The leaves of a Merkle tree, all of the same length, stored contiguously:
/// leaf `i` is `as_flat_slice()[i * leaf_len..(i + 1) * leaf_len]`. One allocation
/// instead of one per leaf, which for a polynomial commitment of 2^19 rows
/// of 139 elements is the difference between a copy and a quarter of a
/// second.
///
/// With `std`, the leaves also dereference to `[Vec<F>]`, so code written for
/// upstream plonky2's `Vec<Vec<F>>` leaves keeps reading them unchanged
/// (`tree.leaves[i]`, `tree.leaves.iter()`, `&tree.leaves[..]`). That view is
/// a full host copy with one vector per leaf, built on first use and kept;
/// plonky2 itself reads rows with [`MerkleLeaves::row`] and
/// [`MerkleLeaves::rows`] and never builds it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MerkleLeaves<F: Copy> {
    values: LeafValues<F>,
    pub leaf_len: usize,
    pub num_leaves: usize,
    #[cfg(feature = "std")]
    #[serde(skip)]
    row_view: RowView<F>,
}

/// The cached `Vec<Vec<F>>` copy behind the read-compatible view. It is a
/// cache, not part of the leaves' value: clones start without it, it never
/// affects equality, and it is not serialized.
#[cfg(feature = "std")]
struct RowView<F>(std::sync::OnceLock<Vec<Vec<F>>>);

#[cfg(feature = "std")]
impl<F> Default for RowView<F> {
    fn default() -> Self {
        Self(std::sync::OnceLock::new())
    }
}

#[cfg(feature = "std")]
impl<F> Clone for RowView<F> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "std")]
impl<F> PartialEq for RowView<F> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

#[cfg(feature = "std")]
impl<F> Eq for RowView<F> {}

#[cfg(feature = "std")]
impl<F> core::fmt::Debug for RowView<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RowView(built: {})", self.0.get().is_some())
    }
}

/// Row views built over device-resident leaves, which only a GPU proof
/// creates; tests assert that proving never builds one.
#[cfg(all(test, feature = "gpu"))]
pub(crate) static DEVICE_ROW_VIEWS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "std")]
impl<F: Copy> core::ops::Deref for MerkleLeaves<F> {
    type Target = [Vec<F>];

    /// The leaves as `Vec<Vec<F>>` rows, for code written against upstream
    /// plonky2. Builds and keeps a full host copy (downloading resident
    /// leaves) on first use.
    fn deref(&self) -> &[Vec<F>] {
        self.row_view.0.get_or_init(|| {
            #[cfg(all(test, feature = "gpu"))]
            if self.values.device.is_some() {
                DEVICE_ROW_VIEWS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
            self.to_rows()
        })
    }
}

impl<F: Copy> Default for MerkleLeaves<F> {
    fn default() -> Self {
        Self {
            values: Vec::new().into(),
            leaf_len: 0,
            num_leaves: 0,
            #[cfg(feature = "std")]
            row_view: RowView::default(),
        }
    }
}

impl<F: Copy> MerkleLeaves<F> {
    /// `num_leaves` leaves of `leaf_len` elements.
    pub fn new(values: Vec<F>, leaf_len: usize, num_leaves: usize) -> Self {
        assert_eq!(values.len(), leaf_len * num_leaves);
        Self {
            values: values.into(),
            leaf_len,
            num_leaves,
            #[cfg(feature = "std")]
            row_view: RowView::default(),
        }
    }

    /// Leaves given one by one; they must all have the same length.
    pub fn from_rows(rows: Vec<Vec<F>>) -> Self {
        let num_leaves = rows.len();
        let leaf_len = rows.first().map_or(0, Vec::len);
        let mut values = Vec::with_capacity(num_leaves * leaf_len);
        for row in &rows {
            assert_eq!(
                row.len(),
                leaf_len,
                "Merkle leaves must have the same length"
            );
            values.extend_from_slice(row);
        }
        Self {
            values: values.into(),
            leaf_len,
            num_leaves,
            #[cfg(feature = "std")]
            row_view: RowView::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.num_leaves
    }

    pub fn is_empty(&self) -> bool {
        self.num_leaves == 0
    }

    /// Read a row. Prepared FRI query rows need only a sparse GPU readback,
    /// and rows prepared for a CPU computation come from the mapped device
    /// buffer; other rows materialize the full host copy on first access.
    /// (Named `row`, not `get`, so the slice view's `get` keeps its meaning.)
    pub fn row(&self, i: usize) -> &[F] {
        assert!(i < self.num_leaves);
        #[cfg(feature = "gpu")]
        if let Some(device) = &self.values.device {
            if let Some(queries) = device.queries.get() {
                if let Some(row) = queries.get(&i) {
                    return row;
                }
            }
            if let Some(rows) = device.rows.get() {
                let k = rows.index[i];
                if k != u32::MAX {
                    let start = k as usize * self.leaf_len;
                    return &rows.data[start..start + self.leaf_len];
                }
            }
        }
        &self.as_flat_slice()[i * self.leaf_len..(i + 1) * self.leaf_len]
    }

    /// Number of field elements, without downloading resident leaves.
    pub fn num_values(&self) -> usize {
        self.leaf_len * self.num_leaves
    }

    /// Full contiguous host data. The first call may download resident leaves;
    /// afterwards their GPU buffer and context handle are released, for all clones.
    /// Serialization and equality also materialize the full host data.
    pub fn as_flat_slice(&self) -> &[F] {
        self.values.as_slice()
    }

    /// Mutable host data. Detaches shared resident storage before mutation so
    /// that clones and previously gathered query rows cannot become inconsistent.
    /// The fixed-length slice preserves the shape of the leaves.
    pub fn as_flat_slice_mut(&mut self) -> &mut [F] {
        // The row view would go stale.
        #[cfg(feature = "std")]
        {
            self.row_view = RowView::default();
        }
        #[cfg(feature = "gpu")]
        if self.values.device.is_some() {
            let values = core::mem::replace(&mut self.values, Vec::new().into());
            self.values = values.into_vec().into();
        }
        &mut self.values.host
    }

    /// Consume the leaves as a contiguous host vector, downloading if needed.
    pub fn into_flat_vec(self) -> Vec<F> {
        self.values.into_vec()
    }

    /// The rows as slices, without building a row copy. (Named `rows`, not
    /// `iter`, so the slice view's `iter` keeps its meaning.)
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[F]> + '_ {
        (0..self.num_leaves).map(|i| self.row(i))
    }

    /// The leaves one by one.
    pub fn to_rows(&self) -> Vec<Vec<F>> {
        self.rows().map(<[F]>::to_vec).collect()
    }
}

/// A tree's digests in Plonky2's flattened layout (see `MerkleTree::digests`).
/// Host data, or, within a GPU proof, a device buffer from which the prover
/// reads only the authentication paths of the FRI queries. Dereferencing,
/// equality and serialization download the full array once, after which the
/// device buffer is released for all clones.
#[derive(Clone, Debug)]
pub struct MerkleDigests<T: Copy> {
    host: Vec<T>,
    #[cfg(feature = "gpu")]
    device: Option<std::sync::Arc<DeviceDigests<T>>>,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct DeviceDigests<T: Copy> {
    resident: std::sync::Mutex<Option<ResidentBuffer>>,
    count: usize,
    decode: fn(&[u8]) -> Vec<T>,
    full: std::sync::OnceLock<Vec<T>>,
    /// Authentication paths by leaf index for the most recently prepared
    /// query set.
    paths: std::sync::Mutex<std::collections::HashMap<usize, Vec<T>>>,
}

/// Decodes 32-byte digests. Never on the Rayon pool: callers hold the
/// resident lock or initialize a `OnceLock`, and a stolen `prove` task on the
/// same thread would wait on them. Large inputs use scoped threads.
#[cfg(feature = "gpu")]
fn decode_hashes<F: RichField, H: Hasher<F>>(bytes: &[u8]) -> Vec<H::Hash> {
    let chunks = bytes.as_chunks::<32>().0;
    if chunks.len() < 1 << 16 {
        return chunks.iter().map(|c| H::Hash::from_bytes(c)).collect();
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 8);
    let chunk_len = chunks.len().div_ceil(workers);
    let parts = std::thread::scope(|scope| {
        chunks
            .chunks(chunk_len)
            .map(|run| {
                scope.spawn(move || {
                    run.iter()
                        .map(|c| H::Hash::from_bytes(c))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("digest decode worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut out = Vec::with_capacity(chunks.len());
    for part in parts {
        out.extend(part);
    }
    out
}

impl<T: Copy> Default for MerkleDigests<T> {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl<T: Copy> From<Vec<T>> for MerkleDigests<T> {
    fn from(host: Vec<T>) -> Self {
        Self {
            host,
            #[cfg(feature = "gpu")]
            device: None,
        }
    }
}

impl<T: Copy> core::ops::Deref for MerkleDigests<T> {
    type Target = Vec<T>;

    /// The full array, as upstream plonky2's `Vec`; downloads resident digests
    /// on first use.
    fn deref(&self) -> &Vec<T> {
        #[cfg(feature = "gpu")]
        if let Some(d) = &self.device {
            return d.full.get_or_init(|| {
                let mut resident = d.resident.lock().unwrap();
                let r = resident
                    .as_ref()
                    .expect("resident digests before materialization");
                log::debug!("gpu: downloading all {} digests to the host", d.count);
                let values =
                    r.context
                        .read_after(&r.buffer, (d.count * 32) as u64, Some(&r.done), d.decode);
                resident.take();
                values
            });
        }
        &self.host
    }
}

impl<T: Copy + PartialEq> PartialEq for MerkleDigests<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl<T: Copy + Eq> Eq for MerkleDigests<T> {}
impl<T: Copy + Serialize> Serialize for MerkleDigests<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(serializer)
    }
}
impl<'de, T: Copy + Deserialize<'de>> Deserialize<'de> for MerkleDigests<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Vec::<T>::deserialize(deserializer)?.into())
    }
}

#[cfg(feature = "gpu")]
impl<T: Copy> MerkleDigests<T> {
    /// `count` digests (32 bytes each) left on the device by submission `done`.
    pub(crate) fn from_device<F: RichField, H: Hasher<F, Hash = T>>(
        context: std::sync::Arc<crate::gpu::GpuContext>,
        buffer: crate::gpu::DeviceBuffer,
        count: usize,
        done: wgpu::SubmissionIndex,
    ) -> Self {
        assert!(buffer.size() >= (count * 32) as u64);
        Self {
            host: Vec::new(),
            device: Some(std::sync::Arc::new(DeviceDigests {
                resident: std::sync::Mutex::new(Some(ResidentBuffer {
                    buffer,
                    context,
                    done,
                })),
                count,
                decode: decode_hashes::<F, H>,
                full: Default::default(),
                paths: Default::default(),
            })),
        }
    }

    /// Gathers the authentication paths of `indices` in a tree over
    /// `leaves_len` leaves with a cap of height `cap_height`, in one
    /// submission and one small readback, so that `MerkleTree::prove` need
    /// not download the full array. Each call replaces the previous set;
    /// other indices fall back to the full download.
    pub(crate) fn prepare_paths(&self, indices: &[usize], leaves_len: usize, cap_height: usize) {
        let Some(d) = &self.device else { return };
        if d.full.get().is_some() || indices.is_empty() {
            return;
        }
        let resident = d.resident.lock().unwrap();
        let Some(r) = resident.as_ref() else { return };
        let mut indices = indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        let depth = log2_strict(leaves_len) - cap_height;
        assert_eq!(d.count, 2 * (leaves_len - (1 << cap_height)));
        assert!(indices.iter().all(|&i| i < leaves_len));
        let paths = if depth == 0 {
            indices.iter().map(|&i| (i, Vec::new())).collect()
        } else {
            let ctx = &r.context;
            let compact = ctx.storage("merkle query siblings", (indices.len() * depth * 32) as u64);
            let mut encoder = ctx.encoder("gather merkle query siblings");
            let tree_len = d.count >> cap_height;
            for (query, &index) in indices.iter().enumerate() {
                // The same sibling positions as `merkle_tree_prove`.
                let tree_start = (index >> depth) * tree_len;
                let mut pair = index & ((1 << depth) - 1);
                for level in 0..depth {
                    let parity = pair & 1;
                    pair >>= 1;
                    let sibling =
                        tree_start + 2 * ((pair << (level + 1)) + (1 << level) - 1) + 1 - parity;
                    encoder.copy_buffer_to_buffer(
                        &r.buffer,
                        sibling as u64 * 32,
                        &compact,
                        ((query * depth + level) * 32) as u64,
                        32,
                    );
                }
            }
            let done = ctx.submit_async(encoder);
            let hashes = ctx.read_after(
                &compact,
                (indices.len() * depth * 32) as u64,
                Some(&done),
                d.decode,
            );
            indices
                .iter()
                .zip(hashes.chunks_exact(depth))
                .map(|(&i, path)| (i, path.to_vec()))
                .collect()
        };
        drop(resident);
        *d.paths.lock().unwrap() = paths;
    }

    /// A prepared authentication path, if `leaf_index` was prepared.
    pub(crate) fn prepared_path(&self, leaf_index: usize) -> Option<Vec<T>> {
        self.device
            .as_ref()?
            .paths
            .lock()
            .unwrap()
            .get(&leaf_index)
            .cloned()
    }

    /// True while a device copy is retained. Does not download any data.
    #[cfg(test)]
    pub(crate) fn is_device_resident(&self) -> bool {
        self.device
            .as_ref()
            .is_some_and(|d| d.resident.lock().unwrap().is_some())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: MerkleLeaves<F>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: MerkleDigests<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: MerkleLeaves::default(),
            digests: MerkleDigests::default(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

/// Fills the digests of the subtree over `leaves`, `leaf_len` elements each, stored contiguously.
pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_len: usize,
) -> H::Hash {
    let num_leaves = leaves
        .len()
        .checked_div(leaf_len)
        .unwrap_or(digests_buf.len() / 2 + 1);
    assert_eq!(num_leaves, digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(&leaves[..leaf_len])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(num_leaves / 2 * leaf_len);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves, leaf_len),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves, leaf_len),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

/// Fills the digests and cap of the tree over `leaves` (see [`MerkleLeaves`]).
pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &MerkleLeaves<F>,
    cap_height: usize,
) {
    let leaf_len = leaves.leaf_len;
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf.par_iter_mut().enumerate().for_each(|(i, cap_buf)| {
            cap_buf.write(H::hash_or_noop(leaves.row(i)));
        });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    digests_chunks
        .zip(cap_buf)
        .enumerate()
        .for_each(|(i, (subtree_digests, subtree_cap))| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            let subtree_leaves = &leaves.as_flat_slice()
                [i * subtree_leaves_len * leaf_len..(i + 1) * subtree_leaves_len * leaf_len];
            subtree_cap.write(fill_subtree::<F, H>(
                subtree_digests,
                subtree_leaves,
                leaf_len,
            ));
        });
}

pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    /// Builds the tree over `leaves`, given one by one (all of the same length), on the GPU
    /// when the `gpu` feature is enabled and a suitable adapter is available, otherwise on
    /// the CPU. Both produce the same tree.
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        Self::from_leaves(MerkleLeaves::from_rows(leaves), cap_height)
    }

    /// `new` for leaves already stored contiguously.
    pub fn from_leaves(leaves: MerkleLeaves<F>, cap_height: usize) -> Self {
        #[cfg(feature = "gpu")]
        if let Some((digests, cap)) = crate::gpu::merkle::try_new::<F, H>(&leaves, cap_height) {
            return Self {
                leaves,
                digests: digests.into(),
                cap: MerkleCap(cap),
            };
        }
        Self::new_cpu(leaves, cap_height)
    }

    /// Builds the tree over `leaves` on the CPU.
    pub(crate) fn new_cpu(leaves: MerkleLeaves<F>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves, cap_height);

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves,
            digests: digests.into(),
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        self.leaves.row(i)
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        #[cfg(feature = "gpu")]
        if let Some(siblings) = self.digests.prepared_path(leaf_index) {
            return MerkleProof { siblings };
        }
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.leaves.len(), cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    /// Code written against upstream plonky2's `Vec<Vec<F>>` leaves and
    /// `Vec<Hash>` digests keeps compiling and reading the same values.
    #[cfg(feature = "std")]
    #[test]
    fn leaves_and_digests_read_like_upstream_fields() -> Result<()> {
        use crate::field::types::Field;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<2>>::F;
        type H = <C as GenericConfig<2>>::Hasher;
        let rows = random_data::<F>(16, 7);
        let tree = MerkleTree::<F, H>::new(rows.clone(), 1);

        // Indexing, length, slicing and iteration as on `Vec<Vec<F>>`.
        assert_eq!(tree.leaves[3], rows[3]);
        assert_eq!(tree.leaves.len(), rows.len());
        let as_slice: &[Vec<F>] = &tree.leaves;
        assert_eq!(as_slice, rows.as_slice());
        let first: Option<&Vec<F>> = tree.leaves.first();
        assert_eq!(first, rows.first());
        assert_eq!(tree.leaves.get(99), None);
        for (leaf, row) in tree.leaves.iter().zip(&rows) {
            assert_eq!(leaf, row);
        }
        assert_eq!(tree.leaves.to_vec(), rows);

        // Digests read as `Vec<Hash>`.
        let digests: &Vec<<H as Hasher<F>>::Hash> = &tree.digests;
        assert_eq!(digests.len(), tree.digests.len());
        assert_eq!(tree.digests[0], digests[0]);

        // The view is a cache: it does not change equality or serialization,
        // and clones start without it.
        let fresh = MerkleTree::<F, H>::new(rows.clone(), 1);
        assert_eq!(tree.leaves, fresh.leaves);
        assert_eq!(
            serde_json::to_string(&tree.leaves)?,
            serde_json::to_string(&fresh.leaves)?
        );
        assert!(tree.leaves.clone().row_view.0.get().is_none());

        // Mutation through the flat slice drops the stale view.
        let mut leaves = tree.leaves.clone();
        assert_eq!(leaves[0], rows[0]);
        leaves.as_flat_slice_mut()[0] = F::ONE;
        assert_eq!(leaves[0][0], F::ONE);
        assert_eq!(leaves[0][1..], rows[0][1..]);
        Ok(())
    }
}
