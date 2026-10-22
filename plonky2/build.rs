//! Fingerprints the compiled witness executor for `plonk::witness_program`.
//!
//! A stored witness program is valid only for the compiler, executor and
//! encoding that produced it. This hashes their sources together with the
//! crate version into `WITNESS_PROGRAM_FORMAT`, so programs survive rebuilds
//! that do not touch these files and are rejected after any change to them.
//! What the generators contribute is checked separately, at load time, by a
//! digest of their operations.

use std::path::Path;
use std::{env, fs};

const SOURCES: &[&str] = &[
    "src/plonk/witness_program/region.rs",
    "src/plonk/witness_program/region/cpu.rs",
    "src/plonk/witness_program/region/plan_bytes.rs",
];

fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn main() {
    let mut hash = fnv1a(
        env::var("CARGO_PKG_VERSION").unwrap().as_bytes(),
        0xcbf29ce484222325,
    );
    for source in SOURCES {
        println!("cargo:rerun-if-changed={source}");
        let bytes = fs::read(source).unwrap_or_else(|e| panic!("reading {source}: {e}"));
        hash = fnv1a(source.as_bytes(), hash);
        hash = fnv1a(&(bytes.len() as u64).to_le_bytes(), hash);
        hash = fnv1a(&bytes, hash);
    }
    println!("cargo:rerun-if-changed=build.rs");
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("witness_program_format.rs");
    fs::write(
        out,
        format!("/// Fingerprint of the witness compiler, executor and encoding sources.\npub(crate) const WITNESS_PROGRAM_FORMAT: u64 = {hash:#018x};\n"),
    )
    .unwrap();
}
