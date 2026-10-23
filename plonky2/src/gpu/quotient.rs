//! The quotient polynomial on the device.
//!
//! Every gate's constraints are traced by [`super::trace`] into a DAG, which
//! is linearized into a bytecode program with register allocation: loads of
//! wires, constants and the public inputs hash, field operations, the
//! gate's selector filter, and "emit" operations that add a filtered
//! constraint, weighted by the right power of each alpha, to the row's
//! accumulators. One small kernel interprets the program for every point of
//! the extended domain, reading the committed evaluations straight from the
//! device copies the GPU commitments keep, and adds the permutation argument
//! terms with hand-written loops. The interpreter avoids compiling large
//! per-circuit shaders: circuits share a kernel specialized by register count,
//! challenge count and quintic usage, with bytecode in a device buffer. A
//! coset IFFT then gives the quotient's coefficients, which the prover splits
//! and commits as usual.

use core::hash::{Hash, Hasher};
use std::sync::Arc;

use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::{Field, Field64};
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::PolynomialBatch;
use crate::gates::selectors::UNUSED_SELECTOR;
use crate::gpu::lde::{fft, FftJob};
use crate::gpu::shader::HASH_WORKGROUP;
use crate::gpu::trace::{
    mds_matched_trace, negation_folded_trace, quintic_fused_trace, trace_gate, GateTrace, Node,
};
use crate::gpu::{
    context, field_supported, module, pipeline, shader, storage_layout, DeviceBuffer, GpuContext,
};
use crate::hash::hash_types::{HashOut, RichField};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::{log2_ceil, log2_strict};

// Opcodes of the program. An op is `[code | dst << 8, y, z, w]`.
const OP_WIRE: u32 = 0;
const OP_CONSTANT: u32 = 1;
const OP_PUBLIC_INPUT: u32 = 2;
/// `dst = Field(y, z)`.
const OP_LITERAL: u32 = 3;
const OP_ADD: u32 = 4;
const OP_SUB: u32 = 5;
const OP_MUL: u32 = 6;
/// `filter = regs[y]`.
const OP_FILTER: u32 = 7;
/// `h[c] += regs[y] * alpha_c^z` for every challenge `c`: a constraint,
/// weighted, added to the gate's accumulators.
const OP_WEIGHTED: u32 = 8;
/// `acc[c] += h[c] * filter` for every challenge `c`, ending a gate.
const OP_FINISH: u32 = 12;
/// `regs[y..y + 12] = mds12(regs[y..y + 12])`: the Poseidon MDS layer in place.
const OP_MDS: u32 = 13;
/// `dst = regs[y]`.
const OP_MOV: u32 = 14;
/// `dst = regs[y]^2` (three widening products in the portable flavor).
const OP_SQUARE: u32 = 15;
/// `dst = regs[y]^7`, the Poseidon S-box, as one op.
const OP_SBOX: u32 = 16;
/// `dst = regs[y] * z` for a constant `z` below 2^32 (one 96-bit reduction).
const OP_MULC: u32 = 9;
/// `dst = regs[y] + Field(z, w)` for a canonical constant.
const OP_ADDC: u32 = 10;
/// `dst = regs[y] * regs[z] + regs[w]` (one reduction).
const OP_MULADD: u32 = 11;
/// `regs[dst..dst + 5] = x * y` in the quintic extension `F[X] / (X^5 - 3)`,
/// where the ten coefficient operands are packed as bytes: `x_0..x_3` in
/// `y`, `x_4, y_0..y_2` in `z`, `y_3, y_4` in `w`. Bit `16 + i` of the first
/// word marks operand `i` as a wire index instead of a register.
const OP_QMUL: u32 = 17;
/// `dst = regs[w] - regs[y] * regs[z]`.
const OP_MULSUB: u32 = 18;
/// `dst = regs[y] * regs[z] - regs[w]`.
const OP_MULRSUB: u32 = 19;

// The opcode byte is the base op in bits 0..5 and, in bits 5..8, one bit per
// operand slot (y, z, w) that names a wire column instead of a register:
// the operand kinds are decided by the opcode, so the kernel reads each
// operand without a branch.
pub(crate) const WIRE_Y: u32 = 1 << 5;
pub(crate) const WIRE_Z: u32 = 1 << 6;
pub(crate) const WIRE_W: u32 = 1 << 7;
pub(crate) const BASE_MASK: u32 = (1 << 5) - 1;

/// Operand slots of a base op that hold a register or a wire.
pub(crate) fn arity(base: u32) -> usize {
    match base {
        OP_ADD | OP_SUB | OP_MUL => 2,
        OP_MULADD | OP_MULSUB | OP_MULRSUB => 3,
        OP_FILTER | OP_WEIGHTED | OP_MULC | OP_ADDC | OP_MOV | OP_SQUARE | OP_SBOX => 1,
        _ => 0,
    }
}

// While a gate is linearized its operand words carry a leaf tag in the top
// nibble; `lower` turns wires into opcode bits, constants and public inputs
// into loads, and `finish_program` literals into preloaded registers.
const SRC_WIRE: u32 = 1 << 28;
const SRC_CONSTANT: u32 = 2 << 28;
const SRC_PUBLIC_INPUT: u32 = 3 << 28;
const SRC_LITERAL: u32 = 4 << 28;
const SRC_MASK: u32 = (1 << 28) - 1;

/// The literals a program refers to, deduplicated.
#[derive(Default, Debug)]
struct Literals {
    table: Vec<u64>,
    index: std::collections::HashMap<u64, u32>,
}

impl Literals {
    fn intern(&mut self, c: u64) -> u32 {
        *self.index.entry(c).or_insert_with(|| {
            self.table.push(c);
            self.table.len() as u32 - 1
        })
    }
}

/// The words of a program buffer.
pub(crate) fn program_words(ops: &[[u32; 4]]) -> Vec<u32> {
    ops.iter().flatten().copied().collect()
}

/// Prepends the loads of the literals into registers `peak..` and points
/// the literal operands at them; returns the register count.
fn finish_program(ops: &mut Vec<[u32; 4]>, literals: &Literals, peak: u32) -> u32 {
    for o in ops.iter_mut() {
        let base = o[0] & BASE_MASK;
        for slot in 1..=arity(base) {
            if o[slot] & !SRC_MASK == SRC_LITERAL {
                o[slot] = peak + (o[slot] & SRC_MASK);
            }
            assert!(o[slot] < SRC_WIRE, "operand still tagged");
        }
    }
    let preload = literals
        .table
        .iter()
        .enumerate()
        .map(|(i, &c)| op(OP_LITERAL, peak + i as u32, c as u32, (c >> 32) as u32, 0));
    ops.splice(0..0, preload);
    peak + literals.table.len() as u32
}

fn op(code: u32, dst: u32, y: u32, z: u32, w: u32) -> [u32; 4] {
    debug_assert!(dst < 1 << 24);
    [code | (dst << 8), y, z, w]
}

/// The largest register file a program may use.
const MAX_REGISTERS: u32 = 1024;
/// Challenges the kernel accumulates.
const MAX_CHALLENGES: usize = 4;

/// The bytecode of a circuit's gate constraints.
#[derive(Debug)]
pub struct QuotientProgram {
    pub ops: DeviceBuffer,
    pub num_ops: u32,
    pub registers: u32,
    /// Whether the finished bytecode needs the quintic interpreter case.
    pub uses_qmul: bool,
    /// The ops as uploaded, for dumps and experiments.
    #[cfg(feature = "gpu-profiling")]
    pub host_ops: Vec<[u32; 4]>,
}

/// A compiled interpreter for a register count and a number of challenges.
#[derive(Debug)]
pub struct QuotientKernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// The identity of a circuit as far as the program is concerned.
fn circuit_key<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for gate in &common_data.gates {
        gate.0.id().hash(&mut hasher);
    }
    common_data
        .selectors_info
        .selector_indices
        .hash(&mut hasher);
    common_data.selectors_info.groups.hash(&mut hasher);
    common_data.num_gate_constraints.hash(&mut hasher);
    common_data.quotient_degree_factor.hash(&mut hasher);
    common_data.num_partial_products.hash(&mut hasher);
    common_data.num_lookup_selectors.hash(&mut hasher);
    common_data.config.num_wires.hash(&mut hasher);
    common_data.config.num_routed_wires.hash(&mut hasher);
    common_data.config.num_constants.hash(&mut hasher);
    common_data.config.num_challenges.hash(&mut hasher);
    hasher.finish()
}

/// Whether a node is loaded or a literal rather than computed.
fn is_leaf(node: &Node) -> bool {
    matches!(
        node,
        Node::Wire(_) | Node::Constant(_) | Node::PublicInput(_) | Node::Const(_)
    )
}

/// The op loading a leaf into `dst`.
fn leaf_op(node: &Node, dst: u32) -> [u32; 4] {
    match *node {
        Node::Wire(i) => op(OP_WIRE, dst, i, 0, 0),
        Node::Constant(j) => op(OP_CONSTANT, dst, j, 0, 0),
        Node::PublicInput(k) => op(OP_PUBLIC_INPUT, dst, k, 0, 0),
        Node::Const(c) => op(OP_LITERAL, dst, c as u32, (c >> 32) as u32, 0),
        _ => unreachable!("not a leaf"),
    }
}

/// The operand word reading a leaf where it is used.
fn leaf_source(node: &Node, literals: &mut Literals) -> u32 {
    let (tag, i) = match *node {
        Node::Wire(i) => (SRC_WIRE, i),
        Node::Constant(j) => (SRC_CONSTANT, j),
        Node::PublicInput(k) => (SRC_PUBLIC_INPUT, k),
        Node::Const(c) => (SRC_LITERAL, literals.intern(c)),
        _ => unreachable!("not a leaf"),
    };
    assert!(i <= SRC_MASK);
    tag | i
}

/// Resolves an instruction's tagged operands: wires become opcode bits,
/// constants and public inputs are loaded into temporaries (released at
/// once: the op reads them before writing its destination), literals keep
/// their tag for `finish_program`.
fn lower(
    code: u32,
    y: u32,
    z: u32,
    w: u32,
    regs: &mut Registers,
    ops: &mut Vec<[u32; 4]>,
) -> (u32, u32, u32, u32) {
    let mut words = [y, z, w];
    let mut code = code;
    let mut temps = Vec::new();
    for (slot, word) in words.iter_mut().enumerate().take(arity(code)) {
        let i = *word & SRC_MASK;
        match *word & !SRC_MASK {
            0 | SRC_LITERAL => {}
            SRC_WIRE => {
                code |= [WIRE_Y, WIRE_Z, WIRE_W][slot];
                *word = i;
            }
            tag => {
                let r = regs.alloc();
                temps.push(r);
                let load = if tag == SRC_CONSTANT {
                    OP_CONSTANT
                } else {
                    OP_PUBLIC_INPUT
                };
                ops.push(op(load, r, i, 0, 0));
                *word = r;
            }
        }
    }
    for r in temps {
        regs.release(r);
    }
    (code, words[0], words[1], words[2])
}

/// The canonical constant of a `Const` node.
fn constant_of(node: &Node) -> Option<u64> {
    match *node {
        Node::Const(c) => Some(c),
        _ => None,
    }
}

/// One step of a gate's program: computing a node, or folding constraint `k`
/// into the gate's Horner accumulators.
#[derive(Clone, Copy)]
enum Event {
    Node(u32),
    Emit(u32),
}

/// Appends the ops of one gate: its filter, then its constraints, each added
/// to the gate's accumulators weighted by `alpha^(weight_offset + k)`, and a
/// final op multiplying the accumulators by the filter into the row's.
/// Returns the register peak, or `None` if packed operands cannot represent
/// this gate. The caller retries without quintic fusion before considering CPU
/// quotient evaluation.
///
/// Each constraint's subtree is evaluated depth first right before it, in
/// constraint order, visiting shared nodes once, which keeps values close to
/// their uses and limits the number of live intermediate values. Computed
/// values get registers released at their last use; leaves are reloaded at
/// each use rather than held.
/// Multiplications by constants below 2^32 and additions of constants use
/// the cheaper ops, and a product used only by one addition is fused into it.
fn append_gate<F: RichField + Field64>(
    ops: &mut Vec<[u32; 4]>,
    literals: &mut Literals,
    trace: &GateTrace<F>,
    selector_index: u32,
    filter_factors: &[u64],
    weight_offset: u32,
) -> Option<u32> {
    // Filter: product of (factor - selector) over the given factors, in registers 0..3.
    ops.push(op(OP_CONSTANT, 0, selector_index, 0, 0));
    match filter_factors {
        [] => ops.push(op(OP_FILTER, 0, SRC_LITERAL | literals.intern(1), 0, 0)),
        [first, rest @ ..] => {
            ops.push(op(OP_SUB, 1, SRC_LITERAL | literals.intern(*first), 0, 0));
            for &factor in rest {
                ops.push(op(OP_SUB, 2, SRC_LITERAL | literals.intern(factor), 0, 0));
                ops.push(op(OP_MUL, 1, 1, 2, 0));
            }
            ops.push(op(OP_FILTER, 0, 1, 0, 0));
        }
    }

    let nodes = &trace.dag.nodes;
    let small_const = |x: u32| constant_of(&nodes[x as usize]).filter(|&c| c < 1 << 32);
    let any_const = |x: u32| constant_of(&nodes[x as usize]).is_some();

    // Evaluation order: the constraints in order, each preceded by the nodes
    // it still needs.
    let mut position = vec![usize::MAX; nodes.len()];
    let mut events: Vec<Event> = Vec::new();
    let mut group_done = vec![false; trace.dag.mds_groups.len()];
    let mut qmul_done = vec![false; trace.dag.qmul_groups.len()];
    for (k, &constraint) in trace.constraints.iter().enumerate() {
        let mut stack = vec![(constraint, false)];
        while let Some((id, expanded)) = stack.pop() {
            if position[id as usize] != usize::MAX {
                continue;
            }
            if expanded || is_leaf(&nodes[id as usize]) {
                position[id as usize] = events.len();
                events.push(Event::Node(id));
                continue;
            }
            stack.push((id, true));
            match nodes[id as usize] {
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    stack.push((b, false));
                    stack.push((a, false));
                }
                Node::Mds(group, _) => {
                    for &input in trace.dag.mds_groups[group as usize].iter().rev() {
                        stack.push((input, false));
                    }
                }
                Node::Qmul(group, _) => {
                    for &input in trace.dag.qmul_groups[group as usize].iter().rev() {
                        stack.push((input, false));
                    }
                }
                _ => {}
            }
        }
        events.push(Event::Emit(k as u32));
    }

    // Uses of each node among the evaluated nodes and the emits; an MDS
    // group uses its inputs once, at its first output.
    let mut uses = vec![0u32; nodes.len()];
    let mut group_first = vec![usize::MAX; trace.dag.mds_groups.len()];
    let mut qmul_first = vec![usize::MAX; trace.dag.qmul_groups.len()];
    for (pos, event) in events.iter().enumerate() {
        match *event {
            Event::Node(id) => match nodes[id as usize] {
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    uses[a as usize] += 1;
                    uses[b as usize] += 1;
                }
                Node::Mds(group, _) if group_first[group as usize] == usize::MAX => {
                    group_first[group as usize] = pos;
                    for input in trace.dag.mds_groups[group as usize] {
                        uses[input as usize] += 1;
                    }
                }
                Node::Qmul(group, _) if qmul_first[group as usize] == usize::MAX => {
                    qmul_first[group as usize] = pos;
                    for input in trace.dag.qmul_groups[group as usize] {
                        uses[input as usize] += 1;
                    }
                }
                _ => {}
            },
            Event::Emit(k) => uses[trace.constraints[k as usize] as usize] += 1,
        }
    }
    let mut fused_into = vec![u32::MAX; nodes.len()];
    // S-boxes: x^7 computed as x^3 * x^4 with x^2 = x * x, x^3 = x * x^2 and
    // x^4 = x^2 * x^2, the intermediate powers used nowhere else.
    let mut sbox_of = vec![u32::MAX; nodes.len()];
    let square_of = |x: u32| match nodes[x as usize] {
        Node::Mul(m0, m1) if m0 == m1 => Some(m0),
        _ => None,
    };
    for event in &events {
        let Event::Node(id) = *event else { continue };
        let Node::Mul(a, b) = nodes[id as usize] else {
            continue;
        };
        for (cube, fourth) in [(a, b), (b, a)] {
            // fourth = s * s, s = x * x, cube = x * s.
            let Some(s) = square_of(fourth) else { continue };
            let Some(x) = square_of(s) else { continue };
            let Node::Mul(c0, c1) = nodes[cube as usize] else {
                continue;
            };
            if !((c0 == x && c1 == s) || (c0 == s && c1 == x)) {
                continue;
            }
            // x^2 is used once by the cube and twice (both operands) by the fourth power.
            if uses[cube as usize] == 1
                && uses[fourth as usize] == 1
                && uses[s as usize] == 3
                && [cube, fourth, s]
                    .iter()
                    .all(|&n| fused_into[n as usize] == u32::MAX)
            {
                for n in [cube, fourth, s] {
                    fused_into[n as usize] = id;
                }
                sbox_of[id as usize] = x;
            }
            break;
        }
    }
    // Products with one use, by an addition, computed by that addition.
    let fusible = |x: u32, uses: &[u32], sbox_of: &[u32]| {
        matches!(nodes[x as usize], Node::Mul(m0, m1) if small_const(m0).is_none() && small_const(m1).is_none())
            && uses[x as usize] == 1
            && sbox_of[x as usize] == u32::MAX
    };
    for event in &events {
        let Event::Node(id) = *event else { continue };
        match nodes[id as usize] {
            Node::Add(a, b) if !any_const(a) && !any_const(b) => {
                if fusible(a, &uses, &sbox_of) {
                    fused_into[a as usize] = id;
                } else if fusible(b, &uses, &sbox_of) {
                    fused_into[b as usize] = id;
                }
            }
            // A subtraction of a constant stays an addition of its negation.
            Node::Sub(a, b) if !any_const(b) => {
                if fusible(b, &uses, &sbox_of) {
                    fused_into[b as usize] = id;
                } else if !any_const(a) && fusible(a, &uses, &sbox_of) {
                    fused_into[a as usize] = id;
                }
            }
            _ => {}
        }
    }

    // Last use of each computed value, by event position; a fused product's
    // operands are used by the addition.
    let mut last_use = vec![0usize; nodes.len()];
    for (pos, event) in events.iter().enumerate() {
        match *event {
            Event::Node(id) => {
                if fused_into[id as usize] != u32::MAX {
                    continue;
                }
                match nodes[id as usize] {
                    Node::Mul(..) if sbox_of[id as usize] != u32::MAX => {
                        let x = sbox_of[id as usize] as usize;
                        last_use[x] = last_use[x].max(pos);
                    }
                    Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                        for x in [a, b] {
                            if fused_into[x as usize] == id {
                                if let Node::Mul(m0, m1) = nodes[x as usize] {
                                    last_use[m0 as usize] = last_use[m0 as usize].max(pos);
                                    last_use[m1 as usize] = last_use[m1 as usize].max(pos);
                                }
                            } else {
                                last_use[x as usize] = last_use[x as usize].max(pos);
                            }
                        }
                    }
                    Node::Mds(group, _) if group_first[group as usize] == pos => {
                        for input in trace.dag.mds_groups[group as usize] {
                            last_use[input as usize] = last_use[input as usize].max(pos);
                        }
                    }
                    Node::Qmul(group, _) if qmul_first[group as usize] == pos => {
                        for input in trace.dag.qmul_groups[group as usize] {
                            last_use[input as usize] = last_use[input as usize].max(pos);
                        }
                    }
                    _ => {}
                }
            }
            Event::Emit(k) => {
                let id = trace.constraints[k as usize] as usize;
                last_use[id] = last_use[id].max(pos);
            }
        }
    }

    let mut register = vec![u32::MAX; nodes.len()];
    let mut regs = Registers::new(3);
    for (pos, event) in events.iter().enumerate() {
        let id = match *event {
            Event::Emit(k) => {
                let id = trace.constraints[k as usize];
                let weight = weight_offset + k;
                if is_leaf(&nodes[id as usize]) {
                    let source = leaf_source(&nodes[id as usize], literals);
                    let (code, y, z, w) = lower(OP_WEIGHTED, source, weight, 0, &mut regs, ops);
                    ops.push(op(code, 0, y, z, w));
                } else {
                    let r = register[id as usize];
                    assert!(r != u32::MAX, "constraint not yet computed");
                    ops.push(op(OP_WEIGHTED, 0, r, weight, 0));
                    if last_use[id as usize] == pos {
                        regs.release(r);
                        register[id as usize] = u32::MAX;
                    }
                }
                continue;
            }
            Event::Node(id) => id,
        };
        let node = &nodes[id as usize];
        if fused_into[id as usize] != u32::MAX || is_leaf(node) {
            continue;
        }
        if let Node::Qmul(group, _) = *node {
            // The first output of a group computes all five coefficients into
            // a block of five registers; the outputs live there. Wire inputs
            // are read in place, other leaves go through temporaries.
            if qmul_done[group as usize] {
                continue;
            }
            qmul_done[group as usize] = true;
            let inputs = trace.dag.qmul_groups[group as usize];
            let mut temps = Vec::new();
            let mut words = [0u32; 10];
            let mut wire_mask = 0u32;
            for (i, &input) in inputs.iter().enumerate() {
                let node = &nodes[input as usize];
                words[i] = match *node {
                    Node::Wire(w) => {
                        wire_mask |= 1 << i;
                        w
                    }
                    _ if is_leaf(node) => {
                        let r = regs.alloc();
                        temps.push(r);
                        ops.push(leaf_op(node, r));
                        r
                    }
                    _ => {
                        let r = register[input as usize];
                        assert!(r != u32::MAX, "quintic product input not yet computed");
                        r
                    }
                };
                if words[i] >= 256 {
                    return None;
                }
            }
            let block = regs.alloc_block(5);
            if block >= 256 {
                return None;
            }
            let packed = |bytes: &[u32]| {
                bytes
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &b)| acc | (b << (8 * i)))
            };
            ops.push([
                OP_QMUL | (block << 8) | (wire_mask << 16),
                packed(&words[0..4]),
                packed(&words[4..8]),
                packed(&words[8..10]),
            ]);
            for &input in &inputs {
                if !is_leaf(&nodes[input as usize])
                    && last_use[input as usize] == pos
                    && register[input as usize] != u32::MAX
                {
                    regs.release(register[input as usize]);
                    register[input as usize] = u32::MAX;
                }
            }
            for r in temps {
                regs.release(r);
            }
            for k in 0..5u32 {
                let output = trace.dag.index_of(Node::Qmul(group, k));
                match output {
                    Some(output)
                        if position[output as usize] != usize::MAX
                            && last_use[output as usize] > pos =>
                    {
                        register[output as usize] = block + k;
                    }
                    _ => regs.release(block + k),
                }
            }
            continue;
        }
        if let Node::Mds(group, _) = *node {
            // The first output of a group computes the whole layer in a block
            // of twelve registers; the outputs live there.
            if group_done[group as usize] {
                continue;
            }
            group_done[group as usize] = true;
            let block = regs.alloc_block(12);
            let inputs = trace.dag.mds_groups[group as usize];
            for (i, &input) in inputs.iter().enumerate() {
                if is_leaf(&nodes[input as usize]) {
                    ops.push(leaf_op(&nodes[input as usize], block + i as u32));
                } else {
                    let r = register[input as usize];
                    assert!(r != u32::MAX, "MDS input not yet computed");
                    ops.push(op(OP_MOV, block + i as u32, r, 0, 0));
                }
            }
            for &input in &inputs {
                if !is_leaf(&nodes[input as usize])
                    && last_use[input as usize] == pos
                    && register[input as usize] != u32::MAX
                {
                    regs.release(register[input as usize]);
                    register[input as usize] = u32::MAX;
                }
            }
            ops.push(op(OP_MDS, 0, block, 0, 0));
            for k in 0..12u32 {
                let output = trace.dag.index_of(Node::Mds(group, k));
                match output {
                    Some(output)
                        if position[output as usize] != usize::MAX
                            && last_use[output as usize] > pos =>
                    {
                        register[output as usize] = block + k;
                    }
                    _ => regs.release(block + k),
                }
            }
            continue;
        }
        let mut temps: Vec<u32> = Vec::new();
        let mut released = Vec::new();
        // The operand word of a node: a leaf read in place, or its register.
        let mut operand = |x: u32,
                           _regs: &mut Registers,
                           _ops: &mut Vec<[u32; 4]>,
                           _temps: &mut Vec<u32>|
         -> u32 {
            if is_leaf(&nodes[x as usize]) {
                leaf_source(&nodes[x as usize], literals)
            } else {
                let r = register[x as usize];
                assert!(r != u32::MAX, "operand not yet computed");
                if last_use[x as usize] == pos {
                    released.push(x);
                }
                r
            }
        };
        let instruction = match *node {
            Node::Mul(a, b) => {
                if sbox_of[id as usize] != u32::MAX {
                    let rx = operand(sbox_of[id as usize], &mut regs, ops, &mut temps);
                    (OP_SBOX, rx, 0, 0)
                } else if let Some(c) = small_const(b) {
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    (OP_MULC, ra, c as u32, 0)
                } else if let Some(c) = small_const(a) {
                    let rb = operand(b, &mut regs, ops, &mut temps);
                    (OP_MULC, rb, c as u32, 0)
                } else if a == b {
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    (OP_SQUARE, ra, 0, 0)
                } else {
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    let rb = operand(b, &mut regs, ops, &mut temps);
                    (OP_MUL, ra, rb, 0)
                }
            }
            Node::Add(a, b) => {
                if let Some(c) = constant_of(&nodes[b as usize]) {
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    (OP_ADDC, ra, c as u32, (c >> 32) as u32)
                } else if let Some(c) = constant_of(&nodes[a as usize]) {
                    let rb = operand(b, &mut regs, ops, &mut temps);
                    (OP_ADDC, rb, c as u32, (c >> 32) as u32)
                } else if fused_into[a as usize] == id || fused_into[b as usize] == id {
                    let (product, other) = if fused_into[a as usize] == id {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    let Node::Mul(m0, m1) = nodes[product as usize] else {
                        unreachable!()
                    };
                    let r0 = operand(m0, &mut regs, ops, &mut temps);
                    let r1 = if m1 == m0 {
                        r0
                    } else {
                        operand(m1, &mut regs, ops, &mut temps)
                    };
                    let rc = operand(other, &mut regs, ops, &mut temps);
                    (OP_MULADD, r0, r1, rc)
                } else {
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    let rb = if b == a {
                        ra
                    } else {
                        operand(b, &mut regs, ops, &mut temps)
                    };
                    (OP_ADD, ra, rb, 0)
                }
            }
            Node::Sub(a, b) => {
                if let Some(c) = constant_of(&nodes[b as usize]) {
                    let neg = (F::ORDER - c) % F::ORDER;
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    (OP_ADDC, ra, neg as u32, (neg >> 32) as u32)
                } else if fused_into[a as usize] == id || fused_into[b as usize] == id {
                    // other - product, or product - other.
                    let (product, other, code) = if fused_into[b as usize] == id {
                        (b, a, OP_MULSUB)
                    } else {
                        (a, b, OP_MULRSUB)
                    };
                    let Node::Mul(m0, m1) = nodes[product as usize] else {
                        unreachable!()
                    };
                    let r0 = operand(m0, &mut regs, ops, &mut temps);
                    let r1 = if m1 == m0 {
                        r0
                    } else {
                        operand(m1, &mut regs, ops, &mut temps)
                    };
                    let rc = operand(other, &mut regs, ops, &mut temps);
                    (code, r0, r1, rc)
                } else {
                    let ra = operand(a, &mut regs, ops, &mut temps);
                    let rb = if b == a {
                        ra
                    } else {
                        operand(b, &mut regs, ops, &mut temps)
                    };
                    (OP_SUB, ra, rb, 0)
                }
            }
            _ => unreachable!(),
        };
        // Lowering may load constants into temporaries: before the operand
        // registers are released, so that a temporary cannot overwrite one.
        let (code, y, z, w) = instruction;
        let (code, y, z, w) = lower(code, y, z, w, &mut regs, ops);
        // The op reads its operands before writing, so registers released
        // here can hold the result.
        for x in released {
            if register[x as usize] != u32::MAX {
                regs.release(register[x as usize]);
                register[x as usize] = u32::MAX;
            }
        }
        for r in temps {
            regs.release(r);
        }
        let dst = regs.alloc();
        ops.push(op(code, dst, y, z, w));
        register[id as usize] = dst;
        if last_use[id as usize] <= pos {
            regs.release(dst);
            register[id as usize] = u32::MAX;
        }
    }
    ops.push(op(OP_FINISH, 0, 0, 0, 0));
    Some(regs.peak)
}

/// Register allocation: single registers from a free list, and blocks of
/// contiguous registers (twelve for the MDS layer, five for a quintic
/// product), reused as blocks of their size once all their registers are
/// released.
struct Registers {
    free: Vec<u32>,
    /// (first register, size) of each free block.
    free_blocks: Vec<(u32, u32)>,
    /// (first register, size, registers still in use) of each block.
    blocks: Vec<(u32, u32, u32)>,
    next: u32,
    peak: u32,
}

impl Registers {
    fn new(reserved: u32) -> Self {
        Self {
            free: Vec::new(),
            free_blocks: Vec::new(),
            blocks: Vec::new(),
            next: reserved,
            peak: reserved,
        }
    }

    fn alloc(&mut self) -> u32 {
        if let Some(r) = self.free.pop() {
            return r;
        }
        self.next += 1;
        self.peak = self.peak.max(self.next);
        self.next - 1
    }

    /// `size` contiguous registers, all in use.
    fn alloc_block(&mut self, size: u32) -> u32 {
        let start = if let Some(i) = self.free_blocks.iter().rposition(|b| b.1 == size) {
            self.free_blocks.swap_remove(i).0
        } else {
            let start = self.next;
            self.next += size;
            self.peak = self.peak.max(self.next);
            self.blocks.push((start, size, 0));
            start
        };
        let block = self.blocks.iter_mut().find(|b| b.0 == start).unwrap();
        block.2 = size;
        start
    }

    fn release(&mut self, r: u32) {
        if let Some(block) = self.blocks.iter_mut().find(|b| r >= b.0 && r < b.0 + b.1) {
            block.2 -= 1;
            if block.2 == 0 {
                self.free_blocks.push((block.0, block.1));
            }
        } else {
            self.free.push(r);
        }
    }
}

/// The program of one gate (its filter over `filter_factors`, its
/// constraints weighted from `weight_offset`), for tests.
#[cfg(test)]
pub(crate) fn gate_program<F: RichField + Field64>(
    trace: &GateTrace<F>,
    selector_index: u32,
    filter_factors: &[u64],
    weight_offset: u32,
) -> Option<(Vec<[u32; 4]>, u32)> {
    let mut ops = Vec::new();
    let mut literals = Literals::default();
    let peak = append_gate(
        &mut ops,
        &mut literals,
        trace,
        selector_index,
        filter_factors,
        weight_offset,
    )?;
    let registers = finish_program(&mut ops, &literals, peak);
    Some((ops, registers))
}

/// Runs a program on the host, for tests: returns the contribution of every
/// challenge `c`: `sum_k filter * alpha_c^(weight_offset + k) * constraint_k`.
#[cfg(test)]
pub(crate) fn run_program_host<F: RichField>(
    ops: &[[u32; 4]],
    registers: usize,
    wires: &[F],
    constants: &[F],
    public_inputs: &[F],
    alphas: &[F],
) -> Vec<F> {
    let mut regs = vec![F::ZERO; registers];
    let mut acc = vec![F::ZERO; alphas.len()];
    let mut h = vec![F::ZERO; alphas.len()];
    let mut filter = F::ONE;
    for op in ops {
        let code = op[0] & BASE_MASK;
        let wires_mask = (op[0] & 0xff) >> 5;
        let dst = (op[0] >> 8) as usize;
        let (y, z, w) = (op[1], op[2], op[3]);
        let lit = |lo: u32, hi: u32| F::from_canonical_u64(lo as u64 | (hi as u64) << 32);
        // Operand of slot 0 (y), 1 (z) or 2 (w).
        let src = |slot: u32, s: u32, regs: &[F]| -> F {
            if wires_mask >> slot & 1 != 0 {
                wires[s as usize]
            } else {
                regs[s as usize]
            }
        };
        match code {
            OP_WIRE => regs[dst] = wires[y as usize],
            OP_CONSTANT => regs[dst] = constants[y as usize],
            OP_PUBLIC_INPUT => regs[dst] = public_inputs[y as usize],
            OP_LITERAL => regs[dst] = lit(y, z),
            OP_ADD => regs[dst] = src(0, y, &regs) + src(1, z, &regs),
            OP_SUB => regs[dst] = src(0, y, &regs) - src(1, z, &regs),
            OP_MUL => regs[dst] = src(0, y, &regs) * src(1, z, &regs),
            OP_FILTER => {
                filter = src(0, y, &regs);
                h.iter_mut().for_each(|h| *h = F::ZERO);
            }
            OP_WEIGHTED => {
                let value = src(0, y, &regs);
                for (h, &alpha) in h.iter_mut().zip(alphas) {
                    *h += value * alpha.exp_u64(z as u64);
                }
            }
            OP_MULC => regs[dst] = src(0, y, &regs) * F::from_canonical_u64(z as u64),
            OP_ADDC => regs[dst] = src(0, y, &regs) + lit(z, w),
            OP_MULADD => regs[dst] = src(0, y, &regs) * src(1, z, &regs) + src(2, w, &regs),
            OP_MULSUB => regs[dst] = src(2, w, &regs) - src(0, y, &regs) * src(1, z, &regs),
            OP_MULRSUB => regs[dst] = src(0, y, &regs) * src(1, z, &regs) - src(2, w, &regs),
            OP_FINISH => {
                for (acc, h) in acc.iter_mut().zip(&h) {
                    *acc += *h * filter;
                }
            }
            OP_MDS => {
                let inputs: [F; 12] = core::array::from_fn(|i| regs[y as usize + i]);
                for k in 0..12 {
                    regs[y as usize + k] = crate::gpu::trace::mds_row(&inputs, k);
                }
            }
            OP_QMUL => {
                let dst = dst & 0xff;
                let wire_mask = op[0] >> 16;
                let inputs: [F; 10] = core::array::from_fn(|i| {
                    let index = ([y, z, w][i / 4] >> (8 * (i % 4))) & 0xff;
                    if wire_mask >> i & 1 != 0 {
                        wires[index as usize]
                    } else {
                        regs[index as usize]
                    }
                });
                for k in 0..5 {
                    regs[dst + k] = crate::gpu::trace::qmul_coeff(&inputs, k);
                }
            }
            OP_MOV => regs[dst] = src(0, y, &regs),
            OP_SQUARE => regs[dst] = src(0, y, &regs).square(),
            OP_SBOX => regs[dst] = src(0, y, &regs).exp_u64(7),
            _ => unreachable!("unknown op {code}"),
        }
    }
    acc
}

/// An unoptimized gate trace and its selector. Preserved across compilation
/// attempts so an encoding limit never forces us to retrace the circuit.
pub(crate) struct TracedGate<F> {
    pub name: String,
    pub trace: GateTrace<F>,
    pub selector_index: u32,
    pub filter_factors: Vec<u64>,
}

pub(crate) struct HostProgram {
    pub ops: Vec<[u32; 4]>,
    pub registers: u32,
    pub uses_qmul: bool,
}

#[derive(Debug)]
enum CompileError {
    PackedQuinticIndexTooLarge,
    UnsupportedLimits,
}

pub(crate) fn uses_qmul(ops: &[[u32; 4]]) -> bool {
    ops.iter().any(|op| op[0] & BASE_MASK == OP_QMUL)
}

fn compile_attempt<F: RichField + Field64>(
    gates: &[TracedGate<F>],
    weight_offset: u32,
    num_challenges: usize,
    fuse_quintics: bool,
) -> Result<HostProgram, CompileError> {
    if num_challenges > MAX_CHALLENGES {
        return Err(CompileError::UnsupportedLimits);
    }
    // Each attempt owns its bytecode, literals and register allocation. A
    // failed fused attempt cannot leave instructions in the unfused program.
    let mut ops = Vec::new();
    let mut literals = Literals::default();
    let mut registers = 0;
    // The trace rewrites before linearization, in this order: written out MDS
    // layers become MDS ops (`trace::mds_matched_trace`), quintic products are
    // fused, then negations are folded (`trace::fold_negations`).
    for gate in gates {
        let matched = mds_matched_trace(&gate.trace);
        let trace = matched.as_ref().unwrap_or(&gate.trace);
        let fused = if fuse_quintics {
            quintic_fused_trace(trace)
        } else {
            None
        };
        let trace = fused.as_ref().unwrap_or(trace);
        let folded = negation_folded_trace(trace);
        let trace = &folded;
        let before = ops.len();
        let peak = append_gate(
            &mut ops,
            &mut literals,
            trace,
            gate.selector_index,
            &gate.filter_factors,
            weight_offset,
        )
        .ok_or(CompileError::PackedQuinticIndexTooLarge)?;
        log::debug!(
            "plonky2 gpu: gate {}: {} ops, {} registers",
            gate.name,
            ops.len() - before,
            peak
        );
        registers = registers.max(peak);
    }
    let working = registers;
    let registers = finish_program(&mut ops, &literals, registers);
    if registers > MAX_REGISTERS {
        return Err(CompileError::UnsupportedLimits);
    }
    log::debug!(
        "plonky2 gpu: quotient registers: {working} working, {} literals",
        literals.table.len()
    );
    Ok(HostProgram {
        uses_qmul: uses_qmul(&ops),
        ops,
        registers: registers.next_power_of_two().max(32),
    })
}

/// Try the fused program first, then ordinary GPU arithmetic if the packed
/// quintic encoding does not fit. Other GPU limits retain the CPU fallback.
pub(crate) fn compile_program<F: RichField + Field64>(
    gates: &[TracedGate<F>],
    weight_offset: u32,
    num_challenges: usize,
) -> Option<HostProgram> {
    match compile_attempt(gates, weight_offset, num_challenges, true) {
        Ok(program) => Some(program),
        Err(CompileError::PackedQuinticIndexTooLarge) => {
            log::debug!("plonky2 gpu: packed quintic indices do not fit; retrying unfused");
            compile_attempt(gates, weight_offset, num_challenges, false).ok()
        }
        Err(CompileError::UnsupportedLimits) => None,
    }
}

/// Builds and uploads a program only after a compilation attempt succeeds.
fn build_program<F: RichField + Extendable<D>, const D: usize>(
    ctx: &GpuContext,
    common_data: &CommonCircuitData<F, D>,
) -> Option<QuotientProgram> {
    let num_selectors = common_data.selectors_info.num_selectors();
    let prefix = num_selectors + common_data.num_lookup_selectors;
    let num_challenges = common_data.config.num_challenges;
    // Terms before the gate constraints: the Z(x) - 1 terms and the partial
    // product checks of every challenge.
    let weight_offset =
        (num_challenges + num_challenges * (common_data.num_partial_products + 1)) as u32;
    let mut gates = Vec::new();
    for (g, gate) in common_data.gates.iter().enumerate() {
        if gate.0.num_constraints() == 0 {
            continue;
        }
        let trace = trace_gate(gate, &common_data.config, prefix)?;
        let selector_index = common_data.selectors_info.selector_indices[g];
        let group = common_data.selectors_info.groups[selector_index].clone();
        let mut filter_factors = group
            .filter(|&i| i != g)
            .map(|i| i as u64)
            .collect::<Vec<_>>();
        if num_selectors > 1 {
            filter_factors.push(UNUSED_SELECTOR as u64);
        }
        gates.push(TracedGate {
            name: gate.0.id(),
            trace,
            selector_index: selector_index as u32,
            filter_factors,
        });
    }
    let HostProgram {
        ops,
        registers,
        uses_qmul,
    } = compile_program(&gates, weight_offset, num_challenges)?;
    let bytes = program_words(&ops)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quotient program"),
        size: (bytes.len() as u64).max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&buffer, 0, &bytes);
    Some(QuotientProgram {
        ops: DeviceBuffer::unpooled(buffer),
        num_ops: ops.len() as u32,
        registers,
        uses_qmul,
        #[cfg(feature = "gpu-profiling")]
        host_ops: ops,
    })
}

/// Change the translated shader entry point for a controlled cache miss,
/// without deleting the user's driver caches (`PLONKY2_GPU_QUOTIENT_NONCE`).
#[cfg(feature = "gpu-profiling")]
fn shader_identity(source: &str) -> (String, String) {
    let entry = crate::util::switches::number::<u64>("PLONKY2_GPU_QUOTIENT_NONCE")
        .map_or_else(|| "quotient".to_owned(), |v| format!("quotient_trial_{v}"));
    (
        source.replace("fn quotient(", &format!("fn {entry}(")),
        entry,
    )
}

const QUINTIC_CASE: &str = r#"
            case 17u: {
                let d = dst & 0xffu;
                let mask = op.x >> 16u;
                var v: array<Field, 10>;
                for (var i = 0u; i < 10u; i++) {
                    var word = op.y;
                    if (i >= 8u) { word = op.w; } else if (i >= 4u) { word = op.z; }
                    let index = (word >> (8u * (i % 4u))) & 0xffu;
                    if ((mask >> i & 1u) != 0u) {
                        v[i] = wire(row, index);
                    } else {
                        v[i] = regs[index];
                    }
                }
                let q = qmul(
                    array<Field, 5>(v[0], v[1], v[2], v[3], v[4]),
                    array<Field, 5>(v[5], v[6], v[7], v[8], v[9]));
                for (var k = 0u; k < 5u; k++) { regs[d + k] = q[k]; }
            }
"#;

const QUOTIENT_KERNEL: &str = r#"
struct QParams {
    // Points of the quotient domain.
    n: u32,
    // Points evaluated by the kernel (the CPU evaluates the rest).
    rows: u32,
    // Rows of the committed evaluations.
    n_lde: u32,
    // Row of the committed evaluations per point of the quotient domain.
    step: u32,
    // Points between a row and the "next" row of the permutation argument.
    next_step: u32,
    // Column of the first sigma polynomial in cs_lde.
    sigmas_offset: u32,
    num_routed_wires: u32,
    max_degree: u32,
    num_prods: u32,
    num_ops: u32,
    // Alpha powers per challenge in chal.
    num_terms: u32,
    // Offset of the beta * k_j table in chal.
    bk_offset: u32,
    // Offset of the alpha power tables in chal.
    alpha_offset: u32,
    pad: u32,
}
@group(0) @binding(0) var<uniform> qp: QParams;
// Constants and sigmas, wires, and Z's and partial products: column-major, n_lde rows.
@group(0) @binding(1) var<storage, read> cs_lde: array<Field>;
@group(0) @binding(2) var<storage, read> w_lde: array<Field>;
@group(0) @binding(3) var<storage, read> z_lde: array<Field>;
// Per point: the shifted point x, L_0(x) and 1 / Z_H(x).
@group(0) @binding(4) var<storage, read> row_table: array<Field>;
// betas, gammas, the public inputs hash, beta_c * k_j, then alpha_c^k for k < num_terms.
@group(0) @binding(5) var<storage, read> chal: array<Field>;
// The program.
@group(0) @binding(6) var<storage, read> ops: array<vec4<u32>>;
// The quotient values, NC columns of n points.
@group(0) @binding(7) var<storage, read_write> out: array<Field>;

fn wire(row: u32, i: u32) -> Field { return w_lde[i * qp.n_lde + row * qp.step]; }
fn constant(row: u32, j: u32) -> Field { return cs_lde[j * qp.n_lde + row * qp.step]; }
fn sigma(row: u32, j: u32) -> Field { return cs_lde[(qp.sigmas_offset + j) * qp.n_lde + row * qp.step]; }
fn zs(row: u32, c: u32) -> Field { return z_lde[c * qp.n_lde + row * qp.step]; }
fn zs_next(row: u32, c: u32) -> Field {
    return z_lde[c * qp.n_lde + ((row + qp.next_step) % qp.n) * qp.step];
}
fn partial_product(row: u32, k: u32) -> Field { return z_lde[(NC + k) * qp.n_lde + row * qp.step]; }
fn alpha_pow(c: u32, k: u32) -> Field { return chal[qp.alpha_offset + c * qp.num_terms + k]; }

@compute @workgroup_size(64)
fn quotient(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= qp.rows) { return; }
    let x = row_table[3u * row];
    let l0 = row_table[3u * row + 1u];
    let zh_inv = row_table[3u * row + 2u];
    var acc: array<Field, NC>;
    // Term k = c: L_0(x) (Z_c(x) - 1).
    for (var c = 0u; c < NC; c++) {
        let term = mul_field(l0, sub_field(zs(row, c), Field(1u, 0u)));
        for (var d = 0u; d < NC; d++) {
            acc[d] = add_field(acc[d], mul_field(term, alpha_pow(d, c)));
        }
    }
    // The partial product checks of every challenge.
    var t = NC;
    let num_chunks = qp.num_prods + 1u;
    for (var c = 0u; c < NC; c++) {
        let beta = chal[c];
        let gamma = chal[NC + c];
        var prev = zs(row, c);
        for (var k = 0u; k < num_chunks; k++) {
            var num_prod = Field(1u, 0u);
            var den_prod = Field(1u, 0u);
            let start = k * qp.max_degree;
            let end = min(start + qp.max_degree, qp.num_routed_wires);
            for (var j = start; j < end; j++) {
                let w = wire(row, j);
                let bk = chal[qp.bk_offset + c * qp.num_routed_wires + j];
                let num = add_field(add_field(w, mul_field(bk, x)), gamma);
                let den = add_field(add_field(w, mul_field(beta, sigma(row, j))), gamma);
                num_prod = mul_field(num_prod, num);
                den_prod = mul_field(den_prod, den);
            }
            var next: Field;
            if (k + 1u == num_chunks) {
                next = zs_next(row, c);
            } else {
                next = partial_product(row, c * qp.num_prods + k);
            }
            let term = sub_field(mul_field(prev, num_prod), mul_field(next, den_prod));
            for (var d = 0u; d < NC; d++) {
                acc[d] = add_field(acc[d], mul_field(term, alpha_pow(d, t)));
            }
            t++;
            prev = next;
        }
    }
    // The gate constraints.
    // BEGIN program
    var regs: array<Field, NREGS>;
    var gate_filter = Field(1u, 0u);
    var h: array<Field, NC>;
    for (var p = 0u; p < qp.num_ops; p++) {
        let op = ops[p];
        let dst = op.x >> 8u;
        switch (op.x & 0xffu) {
            case 0u: { regs[dst] = wire(row, op.y); }
            case 1u: { regs[dst] = constant(row, op.y); }
            case 2u: { regs[dst] = chal[PIH_OFFSET + op.y]; }
            case 3u: { regs[dst] = Field(op.y, op.z); }
            case 12u: {
                for (var d = 0u; d < NC; d++) { acc[d] = mul_add(acc[d], h[d], gate_filter); }
            }
            case 13u: {
                let m = mds12(array<Field, 12>(
                    regs[op.y], regs[op.y + 1u], regs[op.y + 2u], regs[op.y + 3u],
                    regs[op.y + 4u], regs[op.y + 5u], regs[op.y + 6u], regs[op.y + 7u],
                    regs[op.y + 8u], regs[op.y + 9u], regs[op.y + 10u], regs[op.y + 11u]));
                for (var k = 0u; k < 12u; k++) { regs[op.y + k] = m[k]; }
            }
            // QMUL_CASE
            // CASES
            default: {}
        }
    }
    // END program
    for (var c = 0u; c < NC; c++) {
        out[c * qp.n + row] = mul_field(acc[c], zh_inv);
    }
}
"#;

/// The interpreter's WGSL for a register count and a number of challenges.
pub fn kernel_source(
    flavor: shader::Flavor,
    registers: u32,
    num_challenges: u32,
    uses_qmul: bool,
) -> String {
    kernel_source_with_program(flavor, registers, num_challenges, uses_qmul, None)
}

/// The switch cases of the ops with register or wire operands, one per
/// base op and wire-bit combination.
fn interpreter_cases() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for base in [
        OP_ADD,
        OP_SUB,
        OP_MUL,
        OP_FILTER,
        OP_WEIGHTED,
        OP_MULC,
        OP_ADDC,
        OP_MULADD,
        OP_MULSUB,
        OP_MULRSUB,
        OP_MOV,
        OP_SQUARE,
        OP_SBOX,
    ] {
        for mask in 0..1u32 << arity(base) {
            let operand = |slot: u32, word: &str| {
                if mask >> slot & 1 != 0 {
                    format!("wire(row, op.{word})")
                } else {
                    format!("regs[op.{word}]")
                }
            };
            let (y, z, w) = (operand(0, "y"), operand(1, "z"), operand(2, "w"));
            let body = match base {
                OP_ADD => format!("regs[dst] = add_field({y}, {z});"),
                OP_SUB => format!("regs[dst] = sub_field({y}, {z});"),
                OP_MUL => format!("regs[dst] = mul_field({y}, {z});"),
                OP_FILTER => {
                    format!("gate_filter = {y}; for (var d = 0u; d < NC; d++) {{ h[d] = Field(0u, 0u); }}")
                }
                OP_WEIGHTED => format!(
                    "let v = {y}; for (var d = 0u; d < NC; d++) {{ h[d] = mul_add(h[d], v, alpha_pow(d, op.z)); }}"
                ),
                OP_MULC => format!("regs[dst] = mul_small({y}, op.z);"),
                OP_ADDC => format!("regs[dst] = add_const({y}, Field(op.z, op.w));"),
                OP_MULADD => format!("regs[dst] = mul_add({w}, {y}, {z});"),
                OP_MULSUB => format!("regs[dst] = mul_sub({w}, {y}, {z});"),
                OP_MULRSUB => format!("regs[dst] = mul_rsub({w}, {y}, {z});"),
                OP_MOV => format!("regs[dst] = {y};"),
                OP_SQUARE => format!("regs[dst] = square_field({y});"),
                OP_SBOX => format!("regs[dst] = sbox({y});"),
                _ => unreachable!(),
            };
            writeln!(out, "            case {}u: {{ {body} }}", base | mask << 5).unwrap();
        }
    }
    out
}

/// The kernel with the interpreter loop replaced by `program` (WGSL
/// statements over `regs`, `h`, `acc` and `gate_filter`), for experiments
/// with straight-line gate code.
pub fn kernel_source_with_program(
    flavor: shader::Flavor,
    registers: u32,
    num_challenges: u32,
    uses_qmul: bool,
    program: Option<&str>,
) -> String {
    let mut out = shader::prelude_with_mds(flavor);
    if uses_qmul {
        out.push_str(shader::QMUL);
    }
    out.push_str(&format!(
        "const NC: u32 = {num_challenges}u;\nconst NREGS: u32 = {registers}u;\nconst PIH_OFFSET: u32 = {}u;\n",
        2 * num_challenges
    ));
    let kernel = QUOTIENT_KERNEL
        .replace("            // CASES\n", &interpreter_cases())
        .replace(
            "            // QMUL_CASE\n",
            if uses_qmul { QUINTIC_CASE } else { "" },
        );
    match program {
        None => out.push_str(&kernel),
        Some(program) => {
            let begin = kernel.find("    // BEGIN program\n").unwrap();
            let end = kernel.find("    // END program\n").unwrap();
            out.push_str(&kernel[..begin]);
            out.push_str(program);
            out.push_str(&kernel[end..]);
        }
    }
    out
}

impl GpuContext {
    /// The circuit's program, or `None` if it cannot be traced.
    pub(crate) fn quotient_program<F: RichField + Extendable<D>, const D: usize>(
        &self,
        common_data: &CommonCircuitData<F, D>,
    ) -> Option<Arc<QuotientProgram>> {
        let key = circuit_key(common_data);
        let mut cache = self.quotient_programs.lock().unwrap();
        if let Some(program) = cache.get(&key) {
            return program.clone();
        }
        let start = std::time::Instant::now();
        let program = build_program(self, common_data).map(Arc::new);
        match &program {
            Some(program) => log::info!(
                "plonky2 gpu: quotient program for circuit {key:#x}: {} ops, {} registers, built in {:?}",
                program.num_ops,
                program.registers,
                start.elapsed()
            ),
            None => log::info!(
                "plonky2 gpu: circuit {key:#x} cannot be traced, the quotient stays on the CPU"
            ),
        }
        cache.insert(key, program.clone());
        program
    }

    /// The interpreter specialized to the program's register count, challenge
    /// count and use of quintic instructions.
    pub(crate) fn quotient_kernel(
        &self,
        registers: u32,
        num_challenges: u32,
        uses_qmul: bool,
    ) -> Arc<QuotientKernel> {
        let mut cache = self.quotient_kernels.lock().unwrap();
        cache
            .entry((registers, num_challenges, uses_qmul))
            .or_insert_with(|| {
                let source = kernel_source(self.flavor, registers, num_challenges, uses_qmul);
                #[cfg(feature = "gpu-profiling")]
                let (source, entry) = shader_identity(&source);
                #[cfg(not(feature = "gpu-profiling"))]
                let entry = "quotient";
                let layout = storage_layout(
                    &self.device,
                    "quotient",
                    &[
                        (true, true),
                        (false, true),
                        (false, true),
                        (false, true),
                        (false, true),
                        (false, true),
                        (false, true),
                        (false, false),
                    ],
                );
                let module = module(&self.device, "plonky2 quotient", &source);
                let pipeline = pipeline(&self.device, &layout, &module, &entry);
                Arc::new(QuotientKernel { layout, pipeline })
            })
            .clone()
    }

    /// The per-point constants of the quotient domain.
    fn row_table<F: RichField>(
        &self,
        degree_bits: usize,
        quotient_degree_bits: usize,
    ) -> DeviceBuffer {
        let key = (degree_bits as u32, quotient_degree_bits as u32);
        let mut cache = self.row_tables.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                let z_h = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits);
                let points = F::two_adic_subgroup(degree_bits + quotient_degree_bits);
                let shift = F::coset_shift();
                let words = points
                    .par_iter()
                    .enumerate()
                    .flat_map_iter(|(i, &x)| {
                        let shifted_x = shift * x;
                        [
                            shifted_x.to_noncanonical_u64(),
                            z_h.eval_l_0(i, shifted_x).to_noncanonical_u64(),
                            z_h.eval_inverse(i).to_noncanonical_u64(),
                        ]
                    })
                    .collect::<Vec<_>>();
                self.upload("quotient row table", &words)
            })
            .clone()
    }
}

/// Writes the program and the kernel's parameters as text (`PLONKY2_GPU_PROGRAM_DUMP`),
/// unless the file exists: the first line is the `QParams` words, the second
/// the column counts of the three LDE buffers, the challenge buffer length,
/// the number of challenges and the register count, then one op per line.
/// `gpu::tests::benchmarks::bench_quotient_program` reads it.
#[cfg(feature = "gpu-profiling")]
fn dump_program(path: &str, program: &QuotientProgram, params: &[u32], columns: &[u32]) {
    use std::fmt::Write;
    if std::path::Path::new(path).exists() {
        return;
    }
    let mut text = String::new();
    let line = |words: &[u32]| {
        words
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    writeln!(text, "{}", line(params)).unwrap();
    writeln!(text, "{}", line(columns)).unwrap();
    for op in &program.host_ops {
        writeln!(text, "{}", line(op)).unwrap();
    }
    match std::fs::write(path, text) {
        Ok(()) => log::info!("plonky2 gpu: quotient program written to {path}"),
        Err(error) => log::warn!("plonky2 gpu: cannot write {path}: {error}"),
    }
}

/// The device copy of a batch's extended evaluations: the one its GPU
/// commitment kept, or a fresh coset LDE of its polynomials. With `keep`,
/// the buffer stays cached in the batch for later proofs (the circuit's
/// fixed commitment); otherwise it is taken.
fn device_lde<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    ctx: &GpuContext,
    batch: &PolynomialBatch<F, C, D>,
    keep: bool,
) -> DeviceBuffer {
    let cached = if keep {
        batch.device.get()
    } else {
        batch.device.take()
    };
    if let Some(buffer) = cached {
        return buffer;
    }
    let degree = 1usize << batch.degree_log;
    let n = degree << batch.rate_bits;
    let columns = batch.polynomials.len();
    let coeffs = crate::gpu::lde::audit_coefficient_transfer(
        "upload",
        "quotient_lde",
        columns,
        degree,
        || {
            let words = batch
                .polynomials
                .par_iter()
                .flat_map_iter(|p| p.coeffs.iter().map(|x| x.to_noncanonical_u64()))
                .collect::<Vec<_>>();
            ctx.upload("coeffs", &words)
        },
    );
    let lde = ctx.storage("lde", (columns * n * 8) as u64);
    let mut encoder = ctx.encoder("lde");
    fft(
        ctx,
        &mut encoder,
        FftJob {
            log_n: log2_strict(n) as u32,
            columns: columns as u32,
            input: &coeffs,
            in_len: degree as u32,
            in_stride: degree as u32,
            data: &lde,
            output: None,
            shift_in: Some(GoldilocksField::coset_shift()),
            inverse: false,
            shift_out: None,
        },
    );
    ctx.submit(encoder);
    if keep {
        batch.device.set_if_within_limit(ctx, lde.clone());
    }
    lde
}

/// All buffers and dispatches needed by quotient evaluation and its LDEs.
/// Keep this independent of a device so fallback limits can be tested without a GPU.
struct QuotientRequirements {
    log_n: u32,
    log_lde: u32,
    columns: [usize; 3],
    challenges: usize,
    routed_wires: usize,
    terms: usize,
}

impl QuotientRequirements {
    fn fits(&self, max_binding: u64, max_workgroups: u32) -> bool {
        let Some(n) = 1u64.checked_shl(self.log_n) else {
            return false;
        };
        let Some(n_lde) = 1u64.checked_shl(self.log_lde) else {
            return false;
        };
        if n_lde > u32::MAX as u64
            || self.log_n < crate::gpu::MIN_LOG_LEAVES
            || n.div_ceil(HASH_WORKGROUP as u64) > max_workgroups as u64
            || self
                .columns
                .iter()
                .any(|&columns| columns > max_workgroups as usize)
            || shader::fft_plan(self.log_lde)
                .iter()
                .any(|&(_, stages, low_bits)| n_lde >> (stages + low_bits) > max_workgroups as u64)
        {
            return false;
        }
        let challenge_words =
            2 * self.challenges + 4 + self.challenges * (self.routed_wires + self.terms);
        self.columns
            .iter()
            .map(|&columns| (columns as u64, n_lde))
            .chain([
                (3, n),
                (self.challenges as u64, n),
                (challenge_words as u64, 1),
            ])
            .all(|(columns, len)| {
                columns
                    .checked_mul(len)
                    .and_then(|words| words.checked_mul(8))
                    .is_some_and(|bytes| bytes.max(16).next_multiple_of(16) <= max_binding)
            })
    }
}

/// The same eligibility checks are used before commitments and by direct callers.
pub(crate) fn prepare<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
) -> Option<(Arc<GpuContext>, Arc<QuotientProgram>)> {
    if D != 2
        || !field_supported::<F>()
        || common_data.num_lookup_polys != 0
        || !common_data.luts.is_empty()
        || common_data.num_lookup_selectors != 0
    {
        return None;
    }
    let ctx = context()?;
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let rate_bits = common_data.config.fri_config.rate_bits;
    if quotient_degree_bits > rate_bits {
        return None;
    }
    let degree_bits = common_data.degree_bits();
    let log_n = (degree_bits + quotient_degree_bits) as u32;
    let challenges = common_data.config.num_challenges;
    let permutation_columns = challenges * (common_data.num_partial_products + 1);
    let requirements = QuotientRequirements {
        log_n,
        log_lde: (degree_bits + rate_bits) as u32,
        columns: [
            prover_data.constants_sigmas_commitment.polynomials.len(),
            common_data.config.num_wires,
            permutation_columns,
        ],
        challenges,
        routed_wires: common_data.config.num_routed_wires,
        terms: challenges + permutation_columns + common_data.num_gate_constraints,
    };
    if !requirements.fits(ctx.max_binding, ctx.max_workgroups) {
        return None;
    }
    let program = ctx.quotient_program(common_data)?;
    if !ctx.fits(program.num_ops as u64 * 16) {
        return None;
    }
    Some((ctx, program))
}

/// Execute the prepared quotient, or return None for a CPU-quotient plan.
pub(crate) fn try_compute_quotient_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &HashOut<F>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_commitment: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    plan: Option<&crate::gpu::policy::ExecutionPlan>,
) -> Option<Vec<PolynomialCoeffs<F>>> {
    let owned_plan;
    let plan = match plan {
        Some(plan) => plan,
        None => {
            owned_plan = crate::gpu::policy::ExecutionPlan::new(common_data, prover_data);
            &owned_plan
        }
    };
    let (ctx, program) = plan.prepared.as_ref()?;
    let ctx = &**ctx;
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let rate_bits = common_data.config.fri_config.rate_bits;
    let degree_bits = common_data.degree_bits();
    let log_n = (degree_bits + quotient_degree_bits) as u32;
    let n = 1usize << log_n;
    let n_lde = 1usize << (degree_bits + rate_bits);
    let num_challenges = common_data.config.num_challenges;
    let kernel = ctx.quotient_kernel(program.registers, num_challenges as u32, program.uses_qmul);

    let num_routed_wires = common_data.config.num_routed_wires;
    let step = 1 << (rate_bits - quotient_degree_bits);
    let next_step = 1 << quotient_degree_bits;
    let num_terms = num_challenges
        + num_challenges * (common_data.num_partial_products + 1)
        + common_data.num_gate_constraints;

    let start = std::time::Instant::now();
    let cs_lde = device_lde(ctx, &prover_data.constants_sigmas_commitment, true);
    let w_lde = device_lde(ctx, wires_commitment, false);
    let z_lde = device_lde(ctx, zs_commitment, false);
    let row_table = ctx.row_table::<F>(degree_bits, quotient_degree_bits);
    let ldes = start.elapsed();
    let mut chal = Vec::new();
    chal.extend(betas.iter().map(|x| x.to_noncanonical_u64()));
    chal.extend(gammas.iter().map(|x| x.to_noncanonical_u64()));
    chal.extend(
        public_inputs_hash
            .elements
            .iter()
            .map(|x| x.to_noncanonical_u64()),
    );
    let bk_offset = chal.len();
    for &beta in betas {
        chal.extend(
            common_data
                .k_is
                .iter()
                .map(|&k| (beta * k).to_noncanonical_u64()),
        );
    }
    let alpha_offset = chal.len();
    for &alpha in alphas {
        chal.extend(
            alpha
                .powers()
                .take(num_terms)
                .map(|x| x.to_noncanonical_u64()),
        );
    }
    #[cfg(feature = "gpu-profiling")]
    let chal_len = chal.len();
    let chal = ctx.upload("quotient challenges", &chal);
    let out = ctx.storage("quotient values", (num_challenges * n * 8) as u64);
    let cpu_share = plan.cpu_share;
    // With resident leaves on a discrete GPU, CPU participation would
    // download the entire row buffers, so the quotient stays on the device.
    // Where the host maps device buffers directly, the CPU share copies only
    // the rows it reads. Unsupported circuits still fall back to the CPU
    // through the normal, parallel host-materialization path.
    let resident = wires_commitment.merkle_tree.leaves.is_device_resident()
        || zs_commitment.merkle_tree.leaves.is_device_resident();
    let cpu_rows = if cpu_share > 0.0 && (!resident || ctx.mappable) {
        ((n as f64 * cpu_share) as usize).min(n / 2)
    } else {
        0
    };
    let gpu_rows = n - cpu_rows;
    let params_words = [
        n as u32,
        gpu_rows as u32,
        n_lde as u32,
        step,
        next_step,
        common_data.sigmas_range().start as u32,
        num_routed_wires as u32,
        common_data.quotient_degree_factor as u32,
        common_data.num_partial_products as u32,
        program.num_ops,
        num_terms as u32,
        bk_offset as u32,
        alpha_offset as u32,
        0,
    ];
    let params = ctx.uniform("quotient params", &params_words);
    let bindings = ctx.bind(
        &kernel.layout,
        &[
            &params,
            &cs_lde,
            &w_lde,
            &z_lde,
            &row_table,
            &chal,
            &program.ops,
            &out,
        ],
    );
    #[cfg(feature = "gpu-profiling")]
    if let Some(path) = crate::util::switches::text("PLONKY2_GPU_PROGRAM_DUMP") {
        dump_program(
            &path,
            &program,
            &[
                n as u32,
                n as u32,
                n_lde as u32,
                step,
                next_step,
                common_data.sigmas_range().start as u32,
                num_routed_wires as u32,
                common_data.quotient_degree_factor as u32,
                common_data.num_partial_products as u32,
                program.num_ops,
                num_terms as u32,
                bk_offset as u32,
                alpha_offset as u32,
                0,
            ],
            &[
                prover_data.constants_sigmas_commitment.polynomials.len() as u32,
                wires_commitment.polynomials.len() as u32,
                zs_commitment.polynomials.len() as u32,
                chal_len as u32,
                num_challenges as u32,
                program.registers,
            ],
        );
    }
    let coeffs = ctx.storage("quotient coeffs", (num_challenges * n * 8) as u64);
    let work = ctx.storage("quotient ifft work", (num_challenges * n * 8) as u64);
    // The CPU evaluates the last points while the kernel runs, the share
    // following the rates measured on the previous proof.
    let start = std::time::Instant::now();
    #[cfg(feature = "gpu-profiling")]
    let _capture = super::capture::Capture::quotient();
    let (gpu_time, cpu_time) = plonky2_maybe_rayon::join(
        || {
            let start = std::time::Instant::now();
            let mut encoder = ctx.encoder("quotient");
            #[cfg(feature = "gpu-profiling")]
            let span = super::timestamps::ComputeSpan::new(ctx);
            #[cfg(feature = "gpu-profiling")]
            let writes = span.as_ref().map(|t| t.writes(true, true));
            #[cfg(not(feature = "gpu-profiling"))]
            let writes = None;
            {
                let mut pass = ctx.begin_pass(&mut encoder, "quotient", writes);
                pass.set_pipeline(&kernel.pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups((gpu_rows as u32).div_ceil(HASH_WORKGROUP), 1, 1);
            }
            let done = ctx.submit_async(encoder);
            ctx.wait(&done);
            let elapsed = start.elapsed();
            #[cfg(feature = "gpu-profiling")]
            if let Some(span) = span {
                let device_ms = span.read(ctx, &done);
                log::debug!(
                    "GPU_QUOTIENT_TIMING device_ms={device_ms:.6} host_ms={:.6}",
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            elapsed
        },
        || {
            if gpu_rows == n {
                return None;
            }
            let start = std::time::Instant::now();
            // The rows this share reads: each point's own row, and for the
            // Z polynomials also the row `next_step` points ahead.
            wires_commitment.prepare_lde_values(gpu_rows..n, step as usize);
            zs_commitment.prepare_lde_values(
                (gpu_rows..n).flat_map(|i| [i, (i + next_step as usize) % n]),
                step as usize,
            );
            let values = crate::plonk::prover::quotient_values_cpu::<F, C, D>(
                common_data,
                prover_data,
                public_inputs_hash,
                wires_commitment,
                zs_commitment,
                betas,
                gammas,
                &[],
                alphas,
                gpu_rows..n,
            );
            for c in 0..num_challenges {
                let column = values
                    .iter()
                    .map(|row| row[c].to_noncanonical_u64())
                    .collect::<Vec<_>>();
                ctx.write(&out, (c * n + gpu_rows) as u64, &column);
            }
            Some(start.elapsed())
        },
    );
    let evaluation = start.elapsed();
    if let Some(cpu_time) = cpu_time {
        let gpu_rate = gpu_rows as f64 / gpu_time.as_secs_f64().max(1e-6);
        let cpu_rate = (n - gpu_rows) as f64 / cpu_time.as_secs_f64().max(1e-6);
        let concurrent = ctx.options.proving_workers > 1;
        let measured =
            (cpu_rate / (cpu_rate + gpu_rate)).clamp(0.02, if concurrent { 0.4 } else { 0.6 });
        // One worker moves the share to the measured balance;
        // concurrent proofs disturb each other's rates, so they damp it.
        let weight = if concurrent { 0.25 } else { 1.0 };
        let mut share = ctx.quotient_cpu_share.lock().unwrap();
        *share = (1.0 - weight) * *share + weight * measured;
        log::debug!(
            "gpu: quotient: GPU {gpu_rows} points in {gpu_time:?}, CPU {} in {cpu_time:?}, next CPU share {:.3}",
            n - gpu_rows,
            *share
        );
    }
    let start = std::time::Instant::now();
    let mut encoder = ctx.encoder("quotient ifft");
    fft(
        ctx,
        &mut encoder,
        FftJob {
            log_n,
            columns: num_challenges as u32,
            input: &out,
            in_len: n as u32,
            in_stride: n as u32,
            data: &work,
            output: Some(&coeffs),
            shift_in: None,
            inverse: true,
            shift_out: Some(GoldilocksField::coset_shift().inverse()),
        },
    );
    ctx.submit(encoder);
    drop((cs_lde, w_lde, z_lde, out, work));
    Some(crate::gpu::lde::audit_coefficient_transfer(
        "download",
        "quotient_ifft",
        num_challenges,
        n,
        || {
            let words = ctx.read_u64(&coeffs, num_challenges * n);
            log::debug!(
            "gpu: quotient: LDEs and tables {ldes:?}, evaluation {evaluation:?}, IFFT and readback {:?}",
            start.elapsed()
        );
            words
                .par_chunks_exact(n)
                .map(|column| {
                    PolynomialCoeffs::new(
                        column
                            .iter()
                            .map(|&w| F::from_noncanonical_u64(w))
                            .collect(),
                    )
                })
                .collect()
        },
    ))
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    #[test]
    fn quotient_checks_every_buffer_before_selecting_the_gpu() {
        let mut requirements = QuotientRequirements {
            log_n: 17,
            log_lde: 17,
            columns: [82, 135, 20],
            challenges: 2,
            routed_wires: 80,
            terms: 150,
        };
        let binding = 128 << 20;
        // Fixed columns fit, but the wire LDE does not.
        assert!(!requirements.fits(binding, 65535));
        requirements.columns[1] = 100;
        assert!(requirements.fits(binding, 65535));
        requirements.columns[2] = 129;
        assert!(!requirements.fits(binding, 65535));
        requirements.columns = [1; 3];
        // The per-row table is three fields even for a one-column circuit.
        assert!(!requirements.fits(2 << 20, 65535));
        assert!(!requirements.fits(binding, 1));
    }
}
