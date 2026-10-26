//! Byte encoding of compiled whole-circuit programs (see
//! [`crate::plonk::witness_program::WitnessProgram`]).
//!
//! A compiled `RegionProgram` depends only on the circuit (its generators and
//! representative map) and on which targets the prover's inputs set, so a
//! caller can store it and install it in later processes instead of
//! recompiling. The encoding starts with its key: the circuit digest, the
//! generator count, the representative-map length, the sorted input
//! representatives (the mask), the fingerprint of the compiler, executor and
//! encoding sources (`WITNESS_PROGRAM_FORMAT`, from the build script) and a
//! digest of the generators' operations. A checksum covers the payload, so a
//! damaged copy is rejected; the caller compares the key with the circuit and
//! recomputes the operation digest before installing. Both matter because
//! the circuit digest does not cover generator or compiler code: a program
//! from another build could decode cleanly and produce a wrong witness. The
//! operation digest covers generators defined in any crate, and only through
//! what the compiler reads from them. The checksum detects damage, not
//! tampering, so stored bytes are a trusted input, like serialized circuit
//! data.
use core::hash::{Hash, Hasher};

use anyhow::{ensure, Context, Result};
use plonky2_maybe_rayon::*;

use super::{Group, Kind, RegionProgram};
use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::WitnessGeneratorRef;

include!(concat!(env!("OUT_DIR"), "/witness_program_format.rs"));

const MAGIC: &[u8; 8] = b"P2WAOT03";

/// Identity of a whole-circuit program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlanKey {
    pub digest: Vec<u8>,
    pub generators: usize,
    pub map_len: usize,
    /// Sorted, deduplicated representatives of the prover's input targets.
    pub mask: Vec<usize>,
    /// [`WITNESS_PROGRAM_FORMAT`] of the build that compiled the program.
    pub format: u64,
    /// [`operation_digest`] of the circuit's generators.
    pub operations: u128,
    pub compilation: u32,
}

fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deterministic chunked damage checksum. Independent chunks can be checked in
/// parallel, avoiding a serial pass over tens of megabytes during cache load.
/// This retains the format's accidental-damage contract, not authentication.
fn checksum(bytes: &[u8]) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const CHUNK: usize = 128 * 1024;
    let chunks = bytes
        .par_chunks(CHUNK)
        .map(|chunk| fnv1a(chunk, BASIS))
        .collect::<Vec<_>>();
    let mut hash = fnv1a(&(bytes.len() as u64).to_le_bytes(), BASIS);
    for chunk in chunks {
        hash = fnv1a(&chunk.to_le_bytes(), hash);
    }
    hash
}

/// 128-bit FNV-1a, for the operation digest: an identity check against
/// accidental change, not a defence against a chosen collision.
struct Fnv128(u128);

impl Fnv128 {
    fn new() -> Self {
        Self(0x6c62272e07bb014262b821756295c58d)
    }
}

impl Hasher for Fnv128 {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u128;
            self.0 = self.0.wrapping_mul(0x0000000001000000000000000000013b);
        }
    }
    fn finish(&self) -> u64 {
        self.0 as u64
    }
}

/// Digest of everything a compiled program takes from the generators: each
/// generator's complete typed operation and parameters, in generator order
/// (a generator without one counts as absent). Target expansion is a pure
/// function of this descriptor; its definition and Target's mapping semantics
/// are included in WITNESS_PROGRAM_FORMAT. Hashing expanded vectors again would
/// allocate and duplicate the same semantic information on every cache load.
/// Chunks are combined in order independently of the thread count.
pub(crate) fn operation_digest<F: RichField + Extendable<D>, const D: usize>(
    generators: &[WitnessGeneratorRef<F, D>],
) -> u128 {
    const CHUNK: usize = 1 << 13;
    let chunks = generators
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut hasher = Fnv128::new();
            for generator in chunk {
                match generator.0.witness_operation() {
                    None => hasher.write_u8(0),
                    Some(op) => {
                        hasher.write_u8(1);
                        op.hash(&mut hasher);
                    }
                }
            }
            hasher.0
        })
        .collect::<Vec<_>>();
    let mut hasher = Fnv128::new();
    hasher.write_u64(generators.len() as u64);
    for chunk in chunks {
        hasher.write(&chunk.to_le_bytes());
    }
    hasher.0
}

struct Writer(Vec<u8>);

impl Writer {
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    /// `values` as a count, then each value's difference from the previous
    /// one (starting from 0), zigzag-mapped and written as a LEB128 varint.
    /// Index arrays that mostly ascend in small steps, and coefficient
    /// blocks whose lanes repeat a value, shrink to about a byte per value
    /// and compress well.
    fn deltas(&mut self, values: impl ExactSizeIterator<Item = u64>) {
        self.usize(values.len());
        let mut previous = 0u64;
        for value in values {
            let delta = value.wrapping_sub(previous) as i64;
            previous = value;
            let mut v = ((delta << 1) ^ (delta >> 63)) as u64;
            while v >= 0x80 {
                self.0.push(v as u8 | 0x80);
                v >>= 7;
            }
            self.0.push(v as u8);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| anyhow::anyhow!("witness program is truncated"))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).context("witness program value exceeds usize")
    }
    fn len(&mut self, width: usize) -> Result<usize> {
        let n = self.usize()?;
        ensure!(
            n.checked_mul(width)
                .and_then(|bytes| self.pos.checked_add(bytes))
                .is_some_and(|end| end <= self.bytes.len()),
            "witness program is truncated"
        );
        Ok(n)
    }
    /// Decode directly into the destination word type, avoiding a temporary
    /// wide target array and a second conversion pass.
    fn deltas<T: TryFrom<u64>>(&mut self) -> Result<Vec<T>> {
        // Every value takes at least one byte.
        let n = self.len(1)?;
        let mut values = Vec::with_capacity(n);
        let mut previous = 0u64;
        for _ in 0..n {
            let mut v = 0u64;
            let mut shift = 0;
            loop {
                let byte = self.take(1)?[0];
                ensure!(
                    shift < 64 && (shift < 63 || byte <= 1),
                    "witness program varint overflows"
                );
                v |= ((byte & 0x7f) as u64) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            let delta = ((v >> 1) as i64) ^ -((v & 1) as i64);
            previous = previous.wrapping_add(delta as u64);
            values.push(
                T::try_from(previous)
                    .map_err(|_| anyhow::anyhow!("witness program index out of range"))?,
            );
        }
        Ok(values)
    }
}

pub(crate) fn encode(program: &RegionProgram, key: &PlanKey) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(MAGIC);
    w.usize(key.digest.len());
    w.0.extend_from_slice(&key.digest);
    w.usize(key.generators);
    w.usize(key.map_len);
    w.deltas(key.mask.iter().map(|&m| m as u64));
    w.u64(key.format);
    w.u64(key.operations as u64);
    w.u64((key.operations >> 64) as u64);
    w.u32(key.compilation);
    w.usize(program.groups.len());
    for layer in &program.groups {
        w.usize(layer.len());
        for group in layer {
            w.usize(group.kind.code());
            w.u32(group.count);
            w.u32(group.targets_offset);
            w.u32(group.argument.word());
            w.u32(group.inputs);
            w.u32(group.outputs);
        }
    }
    w.deltas(program.targets.iter().map(|&t| t as u64));
    w.deltas(program.coefficients.iter().copied());
    w.deltas(program.representatives.iter().map(|&r| r as u64));
    w.deltas(program.roots.iter().map(|&r| r as u64));
    w.u32(program.random_count);
    w.u32(program.alias_count);
    w.u32(program.alias_pair_offset);
    w.u32(program.state_slots);
    let checksum = checksum(&w.0);
    w.u64(checksum);
    w.0
}

/// Decodes a program and the key it was encoded with. The caller checks the
/// key against the circuit it installs the program on.
pub(crate) fn decode(bytes: &[u8]) -> Result<(PlanKey, RegionProgram)> {
    ensure!(bytes.len() >= 16, "witness program is truncated");
    let (payload, tail) = bytes.split_at(bytes.len() - 8);
    ensure!(
        u64::from_le_bytes(tail.try_into().unwrap()) == checksum(payload),
        "witness program checksum mismatch"
    );
    let mut r = Reader {
        bytes: payload,
        pos: 0,
    };
    ensure!(r.take(8)? == MAGIC, "not a witness program");
    let n = r.len(1)?;
    let key = PlanKey {
        digest: r.take(n)?.to_vec(),
        generators: r.usize()?,
        map_len: r.usize()?,
        mask: r.deltas()?,
        format: r.u64()?,
        operations: r.u64()? as u128 | (r.u64()? as u128) << 64,
        compilation: r.u32()?,
    };
    ensure!(
        key.mask.windows(2).all(|w| w[0] < w[1]) && key.mask.iter().all(|&m| m < key.map_len),
        "witness program input mask is malformed"
    );
    let groups = (0..r.len(8)?)
        .map(|_| {
            (0..r.len(28)?)
                .map(|_| {
                    let kind = Kind::from_code(r.usize()?)?;
                    Ok(Group {
                        kind,
                        count: r.u32()?,
                        targets_offset: r.u32()?,
                        argument: kind.argument(r.u32()?),
                        inputs: r.u32()?,
                        outputs: r.u32()?,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let targets: Vec<u32> = r.deltas()?;
    let coefficients: Vec<u64> = r.deltas()?;
    let representatives: Vec<usize> = r.deltas()?;
    let roots: Vec<usize> = r.deltas()?;
    let random_count = r.u32()?;
    let alias_count = r.u32()?;
    let alias_pair_offset = r.u32()?;
    let state_slots = r.u32()?;
    ensure!(r.pos == payload.len(), "plan file has trailing data");
    ensure!(
        representatives.starts_with(&roots) && representatives.len() as u64 <= state_slots as u64,
        "plan slot layout is inconsistent"
    );
    for &slot in &targets {
        ensure!(slot < state_slots, "plan target slot out of range");
    }
    ensure!(
        alias_pair_offset as u64 + 2 * alias_count as u64 <= targets.len() as u64,
        "plan alias pairs out of range"
    );
    ensure!(
        representatives.iter().all(|&r| r < key.map_len),
        "plan representative out of range"
    );
    // Every random value has exactly one writer, so no blinding value can
    // be dropped by a damaged or edited plan.
    let mut random_seen = vec![false; random_count as usize];
    for group in groups.iter().flatten() {
        ensure!(group.count > 0, "plan group is empty");
        if group.kind != Kind::RandomValue {
            continue;
        }
        for k in 0..group.count as u64 {
            let index = (group.coefficients_offset()? as u64)
                .checked_add(k)
                .and_then(|i| coefficients.get(i as usize).copied())
                .ok_or_else(|| anyhow::anyhow!("plan random coefficient out of range"))?;
            let seen = usize::try_from(index)
                .ok()
                .and_then(|i| random_seen.get_mut(i))
                .ok_or_else(|| anyhow::anyhow!("plan random index out of range"))?;
            ensure!(!*seen, "plan random index written twice");
            *seen = true;
        }
    }
    ensure!(
        random_seen.iter().all(|&seen| seen),
        "plan leaves a random value without a writer"
    );
    let program = RegionProgram {
        groups,
        targets: targets.into(),
        coefficients: coefficients.into(),
        representatives: representatives.into(),
        roots: roots.into(),
        random_count,
        alias_count,
        alias_pair_offset,
        state_slots,
        runtime: Default::default(),
        fast: Default::default(),
        rep_slots: Default::default(),
        cell_slots: Default::default(),
    };
    Ok((key, program))
}

/// Recomputes the trailing checksum of an encoded plan (tests only).
#[cfg(test)]
pub(crate) fn reseal(bytes: &[u8]) -> Vec<u8> {
    let payload = &bytes[..bytes.len() - 8];
    let mut out = payload.to_vec();
    out.extend_from_slice(&checksum(payload).to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::checksum;

    #[test]
    fn checksum_covers_chunk_boundaries_and_is_independent_of_threads() {
        let mut bytes = (0..(256 * 1024 + 7)).map(|i| i as u8).collect::<Vec<_>>();
        let expected = checksum(&bytes);
        #[cfg(feature = "parallel")]
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(|| checksum(&bytes)), expected);
        }
        for index in [0, 128 * 1024 - 1, 128 * 1024, bytes.len() - 1] {
            bytes[index] ^= 1;
            assert_ne!(checksum(&bytes), expected);
            bytes[index] ^= 1;
        }
        assert_ne!(checksum(&bytes[..bytes.len() - 1]), expected);
    }
}
