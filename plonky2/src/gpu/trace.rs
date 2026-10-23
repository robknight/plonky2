//! Symbolic evaluation of gate constraints for code generation.
//!
//! Every gate defines its constraints twice: natively (`eval_unfiltered`
//! and its base field variants) and as circuit operations
//! (`eval_unfiltered_circuit`) for the recursive verifier. The native
//! versions are Rust code over concrete field types and cannot be traced,
//! but the circuit version records every operation as an arithmetic gate
//! slot plus copy constraints in a `CircuitBuilder`. This module runs
//! `eval_unfiltered_circuit` on a scratch builder with symbolic inputs and
//! interprets the recorded gate instances (arithmetic, multiplication and
//! reducing gates) into a DAG of field operations, from which
//! the quotient compiler generates bytecode. Gates whose circuit evaluation uses
//! anything else (witness generators, other gate types) are reported as
//! untraceable and the quotient falls back to the CPU.
//!
//! The quotient is evaluated at base field points, so every value is in
//! the base field: extension targets are traced as pairs of nodes whose
//! second component folds to zero, and the tracer checks that it does.

#[cfg(test)]
use core::fmt::Write;

use hashbrown::HashMap;

use crate::field::extension::Extendable;
use crate::field::types::PrimeField64;
use crate::gates::arithmetic_base::ArithmeticGate;
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::gate::GateRef;
use crate::gates::multiplication_extension::MulExtensionGate;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::gates::reducing::ReducingGate;
use crate::gates::reducing_extension::ReducingExtensionGate;
#[cfg(test)]
use crate::gpu::shader::literal;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::{Poseidon, SPONGE_WIDTH};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CircuitConfig;
use crate::plonk::vars::EvaluationTargets;

pub type NodeId = u32;

/// One field operation, or an input of the constraint evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// `local_wires[i]`.
    Wire(u32),
    /// `local_constants[j]`, counting from the start of the row (selectors included).
    Constant(u32),
    /// `public_inputs_hash.elements[k]`.
    PublicInput(u32),
    /// A canonical field element.
    Const(u64),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    /// Output `k` of the Poseidon MDS layer over the twelve inputs of
    /// `Dag::mds_groups[group]`.
    Mds(u32, u32),
    /// Coefficient `k` of the product of the two degree-5 extension elements
    /// of `Dag::qmul_groups[group]` (five coefficients each) in
    /// `F[X] / (X^5 - QMUL_W)`.
    Qmul(u32, u32),
}

/// The modulus constant of the quintic extension `Node::Qmul` multiplies in
/// (Goldilocks' `Extendable<5>::W`).
pub const QMUL_W: u64 = 3;

/// A hash-consed DAG of field operations with constant folding.
#[derive(Debug, Default)]
pub struct Dag<F> {
    pub nodes: Vec<Node>,
    /// The inputs of each `Node::Mds` group.
    pub mds_groups: Vec<[NodeId; 12]>,
    /// The two operands, five coefficients each, of each `Node::Qmul` group.
    pub qmul_groups: Vec<[NodeId; 10]>,
    index: HashMap<Node, NodeId>,
    _field: core::marker::PhantomData<F>,
}

impl<F: PrimeField64> Dag<F> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            mds_groups: Vec::new(),
            qmul_groups: Vec::new(),
            index: HashMap::new(),
            _field: core::marker::PhantomData,
        }
    }

    /// The twelve outputs of the Poseidon MDS layer over `inputs`.
    pub fn mds(&mut self, inputs: [NodeId; 12]) -> [NodeId; 12] {
        let group = self.mds_groups.len() as u32;
        self.mds_groups.push(inputs);
        core::array::from_fn(|k| self.push(Node::Mds(group, k as u32)))
    }

    /// The five coefficients of the quintic product of `x` and `y`.
    pub fn qmul(&mut self, x: [NodeId; 5], y: [NodeId; 5]) -> [NodeId; 5] {
        let group = self.qmul_groups.len() as u32;
        let mut inputs = [0; 10];
        inputs[..5].copy_from_slice(&x);
        inputs[5..].copy_from_slice(&y);
        self.qmul_groups.push(inputs);
        core::array::from_fn(|k| self.push(Node::Qmul(group, k as u32)))
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        if let Some(&id) = self.index.get(&node) {
            return id;
        }
        let id = self.nodes.len() as NodeId;
        self.nodes.push(node);
        self.index.insert(node, id);
        id
    }

    /// The id of `node`, if it exists.
    pub fn index_of(&self, node: Node) -> Option<NodeId> {
        self.index.get(&node).copied()
    }

    pub fn constant(&mut self, c: F) -> NodeId {
        self.push(Node::Const(c.to_canonical_u64()))
    }

    fn as_const(&self, id: NodeId) -> Option<F> {
        match self.nodes[id as usize] {
            Node::Const(c) => Some(F::from_canonical_u64(c)),
            _ => None,
        }
    }

    pub fn is_zero(&self, id: NodeId) -> bool {
        self.as_const(id).is_some_and(|c| c.is_zero())
    }

    fn is_one(&self, id: NodeId) -> bool {
        self.as_const(id).is_some_and(|c| c.is_one())
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.constant(x + y);
        }
        if self.is_zero(a) {
            return b;
        }
        if self.is_zero(b) {
            return a;
        }
        self.push(Node::Add(a.min(b), a.max(b)))
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.constant(x - y);
        }
        if self.is_zero(b) {
            return a;
        }
        if a == b {
            return self.constant(F::ZERO);
        }
        self.push(Node::Sub(a, b))
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.constant(x * y);
        }
        if self.is_zero(a) || self.is_zero(b) {
            return self.constant(F::ZERO);
        }
        if self.is_one(a) {
            return b;
        }
        if self.is_one(b) {
            return a;
        }
        self.push(Node::Mul(a.min(b), a.max(b)))
    }

    /// Multiplication by a constant.
    pub fn scale(&mut self, a: NodeId, c: F) -> NodeId {
        let c = self.constant(c);
        self.mul(a, c)
    }
}

/// A quadratic extension element as a pair of base field nodes.
type Ext = [NodeId; 2];

impl<F: PrimeField64> Dag<F> {
    fn ext_add(&mut self, a: Ext, b: Ext) -> Ext {
        [self.add(a[0], b[0]), self.add(a[1], b[1])]
    }

    /// Multiplication in `F[X] / (X^2 - w)`.
    fn ext_mul(&mut self, a: Ext, b: Ext, w: F) -> Ext {
        let a0b0 = self.mul(a[0], b[0]);
        let a1b1 = self.mul(a[1], b[1]);
        let w = self.scale(a1b1, w);
        let a0b1 = self.mul(a[0], b[1]);
        let a1b0 = self.mul(a[1], b[0]);
        [self.add(a0b0, w), self.add(a0b1, a1b0)]
    }

    fn ext_scale(&mut self, a: Ext, c: F) -> Ext {
        [self.scale(a[0], c), self.scale(a[1], c)]
    }
}

/// The traced constraints of one gate.
#[derive(Debug)]
pub struct GateTrace<F> {
    pub dag: Dag<F>,
    /// The constraint values, in the gate's order.
    pub constraints: Vec<NodeId>,
}

/// An operation recorded by the scratch builder.
enum Op {
    ArithExt {
        c0: u64,
        c1: u64,
        m0: [Target; 2],
        m1: [Target; 2],
        a: [Target; 2],
        out: [Target; 2],
    },
    MulExt {
        c0: u64,
        m0: [Target; 2],
        m1: [Target; 2],
        out: [Target; 2],
    },
    Arith {
        c0: u64,
        c1: u64,
        m0: Target,
        m1: Target,
        a: Target,
        out: Target,
    },
    /// `accs[i] = accs[i - 1] * alpha + coeffs[i]`, with `accs[-1] = old_acc`.
    ReduceExt {
        alpha: [Target; 2],
        old_acc: [Target; 2],
        coeffs: Vec<[Target; 2]>,
        accs: Vec<[Target; 2]>,
    },
    Reduce {
        alpha: [Target; 2],
        old_acc: [Target; 2],
        coeffs: Vec<Target>,
        accs: Vec<[Target; 2]>,
    },
    /// The Poseidon MDS layer over twelve extension elements.
    PoseidonMds {
        inputs: Vec<[Target; 2]>,
        outputs: Vec<[Target; 2]>,
    },
}

struct UnionFind(HashMap<Target, Target>);

impl UnionFind {
    fn find(&mut self, t: Target) -> Target {
        let mut root = t;
        while let Some(&parent) = self.0.get(&root) {
            if parent == root {
                break;
            }
            root = parent;
        }
        // Path compression.
        let mut current = t;
        while let Some(&parent) = self.0.get(&current) {
            if parent == root || parent == current {
                break;
            }
            self.0.insert(current, root);
            current = parent;
        }
        root
    }

    fn union(&mut self, a: Target, b: Target) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.0.insert(ra, rb);
        }
    }
}

fn ext_targets<const D: usize>(row: usize, range: core::ops::Range<usize>) -> [Target; 2] {
    let t = ExtensionTarget::<D>::from_range(row, range);
    [t.0[0], t.0[1]]
}

/// Traces `gate` with `config`. The gate's constants are
/// `Node::Constant(constants_prefix + j)`. Quintic fusion is a separate pass.
pub fn trace_gate<F: RichField + Extendable<D>, const D: usize>(
    gate: &GateRef<F, D>,
    config: &CircuitConfig,
    constants_prefix: usize,
) -> Option<GateTrace<F>> {
    if D != 2 {
        return None;
    }
    let w = <F as Extendable<D>>::W;
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    let mut dag = Dag::<F>::new();
    let zero = dag.constant(F::ZERO);
    let mut known: HashMap<Target, NodeId> = HashMap::new();

    let wires = builder.add_virtual_extension_targets(config.num_wires);
    let constants = builder.add_virtual_extension_targets(gate.0.num_constants());
    let public_inputs_hash = builder.add_virtual_hash();
    for (i, wire) in wires.iter().enumerate() {
        known.insert(wire.0[0], dag.push(Node::Wire(i as u32)));
        known.insert(wire.0[1], zero);
    }
    for (j, constant) in constants.iter().enumerate() {
        known.insert(
            constant.0[0],
            dag.push(Node::Constant((constants_prefix + j) as u32)),
        );
        known.insert(constant.0[1], zero);
    }
    for (k, element) in public_inputs_hash.elements.iter().enumerate() {
        known.insert(*element, dag.push(Node::PublicInput(k as u32)));
    }

    let outputs = gate.0.eval_unfiltered_circuit(
        &mut builder,
        EvaluationTargets {
            local_constants: &constants,
            local_wires: &wires,
            public_inputs_hash: &public_inputs_hash,
        },
    );

    // Copy constraints merge targets into classes; a class is known once any
    // member is an input, a constant or a resolved gate output.
    let mut classes = UnionFind(HashMap::new());
    for constraint in &builder.copy_constraints {
        classes.union(constraint.pair.0, constraint.pair.1);
    }
    let mut known_roots: HashMap<Target, NodeId> = HashMap::new();
    for (target, node) in known {
        known_roots.entry(classes.find(target)).or_insert(node);
    }
    for (&target, &value) in &builder.targets_to_constants {
        let node = dag.constant(value);
        known_roots.entry(classes.find(target)).or_insert(node);
    }

    let mut ops = Vec::new();
    for (row, instance) in builder.gate_instances.iter().enumerate() {
        let any = instance.gate_ref.0.as_any();
        let constants = &instance.constants;
        if let Some(gate) = any.downcast_ref::<ArithmeticExtensionGate<D>>() {
            for i in 0..gate.num_ops {
                ops.push(Op::ArithExt {
                    c0: constants[0].to_canonical_u64(),
                    c1: constants[1].to_canonical_u64(),
                    m0: ext_targets::<D>(
                        row,
                        ArithmeticExtensionGate::<D>::wires_ith_multiplicand_0(i),
                    ),
                    m1: ext_targets::<D>(
                        row,
                        ArithmeticExtensionGate::<D>::wires_ith_multiplicand_1(i),
                    ),
                    a: ext_targets::<D>(row, ArithmeticExtensionGate::<D>::wires_ith_addend(i)),
                    out: ext_targets::<D>(row, ArithmeticExtensionGate::<D>::wires_ith_output(i)),
                });
            }
        } else if let Some(gate) = any.downcast_ref::<MulExtensionGate<D>>() {
            for i in 0..gate.num_ops {
                ops.push(Op::MulExt {
                    c0: constants[0].to_canonical_u64(),
                    m0: ext_targets::<D>(row, MulExtensionGate::<D>::wires_ith_multiplicand_0(i)),
                    m1: ext_targets::<D>(row, MulExtensionGate::<D>::wires_ith_multiplicand_1(i)),
                    out: ext_targets::<D>(row, MulExtensionGate::<D>::wires_ith_output(i)),
                });
            }
        } else if let Some(gate) = any.downcast_ref::<ArithmeticGate>() {
            for i in 0..gate.num_ops {
                ops.push(Op::Arith {
                    c0: constants[0].to_canonical_u64(),
                    c1: constants[1].to_canonical_u64(),
                    m0: Target::wire(row, ArithmeticGate::wire_ith_multiplicand_0(i)),
                    m1: Target::wire(row, ArithmeticGate::wire_ith_multiplicand_1(i)),
                    a: Target::wire(row, ArithmeticGate::wire_ith_addend(i)),
                    out: Target::wire(row, ArithmeticGate::wire_ith_output(i)),
                });
            }
        } else if let Some(gate) = any.downcast_ref::<ReducingExtensionGate<D>>() {
            ops.push(Op::ReduceExt {
                alpha: ext_targets::<D>(row, ReducingExtensionGate::<D>::wires_alpha()),
                old_acc: ext_targets::<D>(row, ReducingExtensionGate::<D>::wires_old_acc()),
                coeffs: (0..gate.num_coeffs)
                    .map(|i| ext_targets::<D>(row, ReducingExtensionGate::<D>::wires_coeff(i)))
                    .collect(),
                accs: (0..gate.num_coeffs)
                    .map(|i| ext_targets::<D>(row, gate.wires_accs(i)))
                    .collect(),
            });
        } else if let Some(gate) = any.downcast_ref::<ReducingGate<D>>() {
            ops.push(Op::Reduce {
                alpha: ext_targets::<D>(row, ReducingGate::<D>::wires_alpha()),
                old_acc: ext_targets::<D>(row, ReducingGate::<D>::wires_old_acc()),
                coeffs: gate.wires_coeffs().map(|i| Target::wire(row, i)).collect(),
                accs: (0..gate.num_coeffs)
                    .map(|i| ext_targets::<D>(row, gate.wires_accs(i)))
                    .collect(),
            });
        } else if any.downcast_ref::<PoseidonMdsGate<F, D>>().is_some() {
            ops.push(Op::PoseidonMds {
                inputs: (0..SPONGE_WIDTH)
                    .map(|i| ext_targets::<D>(row, PoseidonMdsGate::<F, D>::wires_input(i)))
                    .collect(),
                outputs: (0..SPONGE_WIDTH)
                    .map(|i| ext_targets::<D>(row, PoseidonMdsGate::<F, D>::wires_output(i)))
                    .collect(),
            });
        } else {
            log::debug!(
                "plonky2 gpu: gate {} is not traceable (its circuit uses {})",
                gate.0.id(),
                instance.gate_ref.0.id()
            );
            return None;
        }
    }

    let mut resolved = vec![false; ops.len()];
    let mut progress = true;
    while progress {
        progress = false;
        for (index, op) in ops.iter().enumerate() {
            if resolved[index] {
                continue;
            }
            let lookup = |classes: &mut UnionFind, known: &HashMap<Target, NodeId>, t: Target| {
                known.get(&classes.find(t)).copied()
            };
            let lookup_ext =
                |classes: &mut UnionFind, known: &HashMap<Target, NodeId>, t: [Target; 2]| {
                    Some([lookup(classes, known, t[0])?, lookup(classes, known, t[1])?])
                };
            let define = |classes: &mut UnionFind,
                          known: &mut HashMap<Target, NodeId>,
                          t: Target,
                          node: NodeId| {
                known.entry(classes.find(t)).or_insert(node);
            };
            match op {
                Op::ArithExt {
                    c0,
                    c1,
                    m0,
                    m1,
                    a,
                    out,
                } => {
                    let (Some(m0), Some(m1), Some(a)) = (
                        lookup_ext(&mut classes, &known_roots, *m0),
                        lookup_ext(&mut classes, &known_roots, *m1),
                        lookup_ext(&mut classes, &known_roots, *a),
                    ) else {
                        continue;
                    };
                    let product = dag.ext_mul(m0, m1, w);
                    let first = dag.ext_scale(product, F::from_canonical_u64(*c0));
                    let second = dag.ext_scale(a, F::from_canonical_u64(*c1));
                    let result = dag.ext_add(first, second);
                    define(&mut classes, &mut known_roots, out[0], result[0]);
                    define(&mut classes, &mut known_roots, out[1], result[1]);
                }
                Op::MulExt { c0, m0, m1, out } => {
                    let (Some(m0), Some(m1)) = (
                        lookup_ext(&mut classes, &known_roots, *m0),
                        lookup_ext(&mut classes, &known_roots, *m1),
                    ) else {
                        continue;
                    };
                    let product = dag.ext_mul(m0, m1, w);
                    let result = dag.ext_scale(product, F::from_canonical_u64(*c0));
                    define(&mut classes, &mut known_roots, out[0], result[0]);
                    define(&mut classes, &mut known_roots, out[1], result[1]);
                }
                Op::Arith {
                    c0,
                    c1,
                    m0,
                    m1,
                    a,
                    out,
                } => {
                    let (Some(m0), Some(m1), Some(a)) = (
                        lookup(&mut classes, &known_roots, *m0),
                        lookup(&mut classes, &known_roots, *m1),
                        lookup(&mut classes, &known_roots, *a),
                    ) else {
                        continue;
                    };
                    let product = dag.mul(m0, m1);
                    let first = dag.scale(product, F::from_canonical_u64(*c0));
                    let second = dag.scale(a, F::from_canonical_u64(*c1));
                    let result = dag.add(first, second);
                    define(&mut classes, &mut known_roots, *out, result);
                }
                Op::ReduceExt {
                    alpha,
                    old_acc,
                    coeffs,
                    accs,
                } => {
                    let (Some(alpha), Some(mut acc)) = (
                        lookup_ext(&mut classes, &known_roots, *alpha),
                        lookup_ext(&mut classes, &known_roots, *old_acc),
                    ) else {
                        continue;
                    };
                    let Some(coeffs) = coeffs
                        .iter()
                        .map(|c| lookup_ext(&mut classes, &known_roots, *c))
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    for (coeff, acc_wires) in coeffs.into_iter().zip(accs) {
                        let product = dag.ext_mul(acc, alpha, w);
                        acc = dag.ext_add(product, coeff);
                        define(&mut classes, &mut known_roots, acc_wires[0], acc[0]);
                        define(&mut classes, &mut known_roots, acc_wires[1], acc[1]);
                    }
                }
                Op::Reduce {
                    alpha,
                    old_acc,
                    coeffs,
                    accs,
                } => {
                    let (Some(alpha), Some(mut acc)) = (
                        lookup_ext(&mut classes, &known_roots, *alpha),
                        lookup_ext(&mut classes, &known_roots, *old_acc),
                    ) else {
                        continue;
                    };
                    let Some(coeffs) = coeffs
                        .iter()
                        .map(|c| lookup(&mut classes, &known_roots, *c))
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    for (coeff, acc_wires) in coeffs.into_iter().zip(accs) {
                        let product = dag.ext_mul(acc, alpha, w);
                        acc = dag.ext_add(product, [coeff, zero]);
                        define(&mut classes, &mut known_roots, acc_wires[0], acc[0]);
                        define(&mut classes, &mut known_roots, acc_wires[1], acc[1]);
                    }
                }
                Op::PoseidonMds { inputs, outputs } => {
                    let Some(inputs) = inputs
                        .iter()
                        .map(|t| lookup_ext(&mut classes, &known_roots, *t))
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    if inputs.iter().all(|input| dag.is_zero(input[1])) {
                        // Base field inputs: one node group, evaluated as a whole.
                        let results = dag.mds(core::array::from_fn(|i| inputs[i][0]));
                        for (r, out) in outputs.iter().enumerate() {
                            define(&mut classes, &mut known_roots, out[0], results[r]);
                            define(&mut classes, &mut known_roots, out[1], zero);
                        }
                        resolved[index] = true;
                        progress = true;
                        continue;
                    }
                    for (r, out) in outputs.iter().enumerate() {
                        let mut acc = [zero, zero];
                        for i in 0..SPONGE_WIDTH {
                            let coeff = F::from_canonical_u64(<F as Poseidon>::MDS_MATRIX_CIRC[i]);
                            let term = dag.ext_scale(inputs[(i + r) % SPONGE_WIDTH], coeff);
                            acc = dag.ext_add(acc, term);
                        }
                        let diag = F::from_canonical_u64(<F as Poseidon>::MDS_MATRIX_DIAG[r]);
                        let term = dag.ext_scale(inputs[r], diag);
                        acc = dag.ext_add(acc, term);
                        define(&mut classes, &mut known_roots, out[0], acc[0]);
                        define(&mut classes, &mut known_roots, out[1], acc[1]);
                    }
                }
            }
            resolved[index] = true;
            progress = true;
        }
    }

    let mut constraints = Vec::with_capacity(outputs.len());
    for output in outputs {
        let re = known_roots.get(&classes.find(output.0[0])).copied();
        let im = known_roots.get(&classes.find(output.0[1])).copied();
        match (re, im) {
            (Some(re), Some(im)) if dag.is_zero(im) => constraints.push(re),
            _ => {
                log::debug!(
                    "plonky2 gpu: gate {} is not traceable (an output is unresolved or not in the base field)",
                    gate.0.id()
                );
                return None;
            }
        }
    }
    if constraints.len() != gate.0.num_constraints() {
        return None;
    }
    Some(GateTrace { dag, constraints })
}

fn lookup_mul_raw<F: PrimeField64>(dag: &Dag<F>, a: NodeId, b: NodeId) -> Option<NodeId> {
    dag.index_of(Node::Mul(a.min(b), a.max(b)))
}

/// Replaces quintic extension products by `Node::Qmul` groups. Each coefficient
/// `k` must be a fold of the plain products `x_i y_(k-i)` plus a fold of the
/// products `x_i y_(5+k-i)` scaled by `QMUL_W`. The anchor is coefficient 0, an
/// addition of one plain product and a four-term chain of scaled products
/// whose factors are the operand pairs `(x_i, y_(5-i))`; every assignment of
/// the pairs to `i` and every orientation of each pair is tried, and one is
/// accepted only if all five coefficient nodes rebuilt from it exist, so a
/// group computes exactly the nodes it replaces.
#[cfg(test)]
pub(crate) fn fuse_quintic_products<F: PrimeField64>(trace: &mut GateTrace<F>) {
    if let Some(fused) = quintic_fused_trace(trace) {
        *trace = fused;
    }
}

/// Returns a fused trace only if a match exists, preserving the original for
/// retry and avoiding a full DAG clone for gates without quintic arithmetic.
pub(crate) fn quintic_fused_trace<F: PrimeField64>(trace: &GateTrace<F>) -> Option<GateTrace<F>> {
    let dag = &trace.dag;
    let nodes = &dag.nodes;
    let Some(w) = dag.index_of(Node::Const(QMUL_W)) else {
        return None;
    };
    let is_const = |id: NodeId| matches!(nodes[id as usize], Node::Const(_));
    // A product with at least one non-constant operand (an operand may be a
    // constant coordinate of the generator point).
    let plain = |id: NodeId| match nodes[id as usize] {
        Node::Mul(a, b) if !(is_const(a) && is_const(b)) => Some((a, b)),
        _ => None,
    };
    // A plain product scaled by w: `Mul(product, w)`, or, when the builder
    // folded a constant operand into the gate constant, `Mul(x, Const(w c))`
    // standing for `w * Mul(x, Const(c))`. Returns the plain product's factors.
    let scaled_const = |x: NodeId, c: u64| -> Option<(NodeId, NodeId)> {
        let c = F::from_canonical_u64(c) * F::from_canonical_u64(QMUL_W).inverse();
        let c = dag.index_of(Node::Const(c.to_canonical_u64()))?;
        lookup_mul_raw(dag, x, c).map(|_| (x, c))
    };
    let scaled = |id: NodeId| -> Option<(NodeId, NodeId)> {
        match nodes[id as usize] {
            Node::Mul(a, b) if b == w && a != w => plain(a),
            Node::Mul(a, b) if a == w && b != w => plain(b),
            Node::Mul(a, b) if !is_const(a) && is_const(b) => match nodes[b as usize] {
                Node::Const(c) => scaled_const(a, c),
                _ => None,
            },
            Node::Mul(a, b) if is_const(a) && !is_const(b) => match nodes[a as usize] {
                Node::Const(c) => scaled_const(b, c),
                _ => None,
            },
            _ => None,
        }
    };
    // The summands of a fold of additions whose summands are never additions
    // themselves (the fold's inner sums are, so they are told apart by kind:
    // the tracer resolves ops in the builder's row order, which groups ops
    // by gate constant, so node ids do not follow the fold).
    let flatten = |id: NodeId| -> Option<Vec<NodeId>> {
        let is_add = |x: NodeId| matches!(nodes[x as usize], Node::Add(..));
        let mut terms = Vec::new();
        let mut id = id;
        loop {
            let Node::Add(a, b) = nodes[id as usize] else {
                terms.push(id);
                break;
            };
            match (is_add(a), is_add(b)) {
                (true, true) => return None,
                (true, false) => {
                    terms.push(b);
                    id = a;
                }
                (false, true) => {
                    terms.push(a);
                    id = b;
                }
                (false, false) => {
                    terms.push(a);
                    terms.push(b);
                    break;
                }
            }
        }
        Some(terms)
    };
    let lookup_mul = |a: NodeId, b: NodeId| lookup_mul_raw(dag, a, b);
    let lookup_add = |a: NodeId, b: NodeId| dag.index_of(Node::Add(a.min(b), a.max(b)));
    // `w * (a * b)` as the tracer built it: a constant operand was folded into
    // the gate constant by the builder, so the product is `Mul(other, Const(w c))`.
    let lookup_scaled = |a: NodeId, b: NodeId| -> Option<NodeId> {
        let folded = |x: NodeId, c: u64| {
            let c = F::from_canonical_u64(c) * F::from_canonical_u64(QMUL_W);
            let c = dag.index_of(Node::Const(c.to_canonical_u64()))?;
            lookup_mul(x, c)
        };
        match (nodes[a as usize], nodes[b as usize]) {
            (Node::Const(_), Node::Const(_)) => None,
            (_, Node::Const(c)) => folded(a, c),
            (Node::Const(c), _) => folded(b, c),
            _ => lookup_mul(lookup_mul(a, b)?, w),
        }
    };
    // The coefficient nodes of x * y as the tracer would have built them.
    let rebuild = |x: &[NodeId; 5], y: &[NodeId; 5]| -> Option<[NodeId; 5]> {
        let fold = |terms: Vec<Option<NodeId>>| -> Option<NodeId> {
            let mut terms = terms.into_iter();
            let mut sum = terms.next()??;
            for term in terms {
                sum = lookup_add(sum, term?)?;
            }
            Some(sum)
        };
        let mut out = [0; 5];
        for (k, slot) in out.iter_mut().enumerate() {
            let term1 = fold((0..=k).map(|i| lookup_mul(x[i], y[k - i])).collect())?;
            if k == 4 {
                *slot = term1;
                continue;
            }
            let term2 = fold(
                (k + 1..5)
                    .map(|i| lookup_scaled(x[i], y[5 + k - i]))
                    .collect(),
            )?;
            *slot = lookup_add(term1, term2)?;
        }
        Some(out)
    };
    // group of each replaced node, and the groups' operands and outputs.
    let mut claimed: HashMap<NodeId, (usize, usize)> = HashMap::new();
    let mut groups: Vec<([NodeId; 10], [NodeId; 5])> = Vec::new();
    // PLONKY2_GPU_QMUL_DEBUG=1 reports the anchors and groups of every traced
    // gate, and PLONKY2_GPU_QMUL_DUMP=<path> writes each gate's DAG and groups
    // to <path>.<node count>.
    #[cfg(feature = "gpu-profiling")]
    let mut anchors = 0usize;
    for id in 0..nodes.len() as NodeId {
        let Node::Add(a, b) = nodes[id as usize] else {
            continue;
        };
        if claimed.contains_key(&id) {
            continue;
        }
        let candidates = [(a, b), (b, a)];
        'candidates: for (p, chain) in candidates {
            let Some((p0, p1)) = plain(p) else { continue };
            let Some(terms) = flatten(chain) else {
                continue;
            };
            if terms.len() != 4 {
                continue;
            }
            let mut pairs = [(0, 0); 4];
            for (pair, &term) in pairs.iter_mut().zip(&terms) {
                let Some(factors) = scaled(term) else {
                    continue 'candidates;
                };
                *pair = factors;
            }
            #[cfg(feature = "gpu-profiling")]
            {
                anchors += 1;
            }
            // Permutations of the four pairs times orientations of the five.
            let permutations = [
                [0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
                [0, 3, 2, 1],
                [1, 0, 2, 3],
                [1, 0, 3, 2],
                [1, 2, 0, 3],
                [1, 2, 3, 0],
                [1, 3, 0, 2],
                [1, 3, 2, 0],
                [2, 0, 1, 3],
                [2, 0, 3, 1],
                [2, 1, 0, 3],
                [2, 1, 3, 0],
                [2, 3, 0, 1],
                [2, 3, 1, 0],
                [3, 0, 1, 2],
                [3, 0, 2, 1],
                [3, 1, 0, 2],
                [3, 1, 2, 0],
                [3, 2, 0, 1],
                [3, 2, 1, 0],
            ];
            for (permutation, orientation) in permutations
                .iter()
                .flat_map(|p| (0..32u32).map(move |o| (p, o)))
            {
                let pick = |bit: u32, (u, v): (NodeId, NodeId)| {
                    if orientation >> bit & 1 == 0 {
                        (u, v)
                    } else {
                        (v, u)
                    }
                };
                let (x0, y0) = pick(0, (p0, p1));
                let mut x = [x0; 5];
                let mut y = [y0; 5];
                for i in 1..5 {
                    let (xi, yj) = pick(i as u32, pairs[permutation[i - 1]]);
                    x[i] = xi;
                    y[5 - i] = yj;
                }
                let Some(outputs) = rebuild(&x, &y) else {
                    continue;
                };
                if outputs[0] != id || outputs.iter().any(|o| claimed.contains_key(o)) {
                    continue;
                }
                let mut inputs = [0; 10];
                inputs[..5].copy_from_slice(&x);
                inputs[5..].copy_from_slice(&y);
                for (k, &o) in outputs.iter().enumerate() {
                    claimed.insert(o, (groups.len(), k));
                }
                groups.push((inputs, outputs));
                break 'candidates;
            }
        }
    }
    #[cfg(feature = "gpu-profiling")]
    {
        if crate::util::switches::flag("PLONKY2_GPU_QMUL_DEBUG") {
            log::debug!("qmul: {} anchors, {} groups", anchors, groups.len());
        }
        if let Some(path) = crate::util::switches::text("PLONKY2_GPU_QMUL_DUMP") {
            use std::io::Write as _;
            let path = format!("{path}.{}", nodes.len());
            let result = (|| -> std::io::Result<()> {
                let mut f = std::fs::File::create(&path)?;
                for (id, node) in nodes.iter().enumerate() {
                    writeln!(f, "{id} {node:?}")?;
                }
                for (inputs, outputs) in &groups {
                    writeln!(f, "group {inputs:?} {outputs:?}")?;
                }
                Ok(())
            })();
            if let Err(error) = result {
                log::warn!("plonky2 gpu: cannot write DAG dump {path}: {error}");
            }
        }
    }
    if groups.is_empty() {
        return None;
    }
    // Rebuild the DAG with the groups' outputs replaced; the inputs of every
    // output precede it, so a group is created at its first output.
    let mut new = Dag::<F>::new();
    let mut map = vec![NodeId::MAX; nodes.len()];
    let mut mds_done = vec![false; dag.mds_groups.len()];
    new.mds_groups = vec![[0; 12]; dag.mds_groups.len()];
    let mut qmul_of = vec![[NodeId::MAX; 5]; groups.len()];
    for (id, node) in nodes.iter().enumerate() {
        let m = |x: NodeId| map[x as usize];
        if let Some(&(g, k)) = claimed.get(&(id as NodeId)) {
            if qmul_of[g][0] == NodeId::MAX {
                let inputs = groups[g].0.map(m);
                assert!(inputs.iter().all(|&i| i != NodeId::MAX));
                let mut x = [0; 5];
                let mut y = [0; 5];
                x.copy_from_slice(&inputs[..5]);
                y.copy_from_slice(&inputs[5..]);
                qmul_of[g] = new.qmul(x, y);
            }
            map[id] = qmul_of[g][k];
            continue;
        }
        map[id] = match *node {
            Node::Add(a, b) => new.push(Node::Add(m(a), m(b))),
            Node::Sub(a, b) => new.push(Node::Sub(m(a), m(b))),
            Node::Mul(a, b) => new.push(Node::Mul(m(a), m(b))),
            Node::Mds(group, k) => {
                if !mds_done[group as usize] {
                    mds_done[group as usize] = true;
                    new.mds_groups[group as usize] = dag.mds_groups[group as usize].map(m);
                }
                new.push(Node::Mds(group, k))
            }
            leaf => new.push(leaf),
        };
    }
    Some(GateTrace {
        constraints: trace.constraints.iter().map(|&c| map[c as usize]).collect(),
        dag: new,
    })
}

/// Visits the assignments of distinct columns to items `i..`, item `i`
/// taking one of `options[i]`, until `visit` returns `Some`.
fn bijections<T>(
    options: &[Vec<usize>],
    i: usize,
    column: &mut [usize],
    used: &mut [bool],
    visit: &mut dyn FnMut(&[usize]) -> Option<T>,
) -> Option<T> {
    if i == options.len() {
        return visit(column);
    }
    for &j in &options[i] {
        if used[j] {
            continue;
        }
        used[j] = true;
        column[i] = j;
        let result = bijections(options, i + 1, column, used, visit);
        used[j] = false;
        if result.is_some() {
            return result;
        }
    }
    None
}

/// Rebuilds a trace with every Poseidon MDS layer that a gate writes out as
/// sums of scaled inputs replaced by `Node::Mds` groups, or returns `None`
/// if there is none.
///
/// Each sum's linear form over its inputs (nodes other than additions and
/// products by constants) is computed up to `SPONGE_WIDTH` terms. Forms with
/// exactly `SPONGE_WIDTH` inputs whose coefficients are entries of the MDS
/// matrix (taken from `mds_row`) are grouped by input set; for a group, an
/// assignment of inputs to matrix columns is accepted only if every form in
/// it is then exactly a row of the matrix, and the group is used only if it
/// covers at least `MIN_MDS_ROWS` distinct rows, since one MDS op costs about
/// as much as two rows written out.
pub(crate) fn mds_matched_trace<F: RichField>(trace: &GateTrace<F>) -> Option<GateTrace<F>> {
    const WIDTH: usize = SPONGE_WIDTH;
    const MIN_MDS_ROWS: usize = 3;
    let dag = &trace.dag;
    let nodes = &dag.nodes;
    let matrix: [[u64; WIDTH]; WIDTH] = core::array::from_fn(|r| {
        core::array::from_fn(|j| {
            let mut unit = [F::ZERO; WIDTH];
            unit[j] = F::ONE;
            mds_row(&unit, r).to_canonical_u64()
        })
    });
    let is_entry = |c: u64| matrix.iter().flatten().any(|&m| m == c);
    // Linear forms: (input, canonical coefficient), sorted by input.
    let mut forms: Vec<Option<Vec<(NodeId, u64)>>> = Vec::with_capacity(nodes.len());
    let constant = |id: NodeId| match nodes[id as usize] {
        Node::Const(c) => Some(c),
        _ => None,
    };
    for (id, node) in nodes.iter().enumerate() {
        let form = match *node {
            Node::Add(a, b) => match (&forms[a as usize], &forms[b as usize]) {
                (Some(x), Some(y)) => {
                    let mut sum: Vec<(NodeId, u64)> = Vec::with_capacity(x.len() + y.len());
                    let (mut i, mut j) = (0, 0);
                    while i < x.len() || j < y.len() {
                        if j == y.len() || (i < x.len() && x[i].0 < y[j].0) {
                            sum.push(x[i]);
                            i += 1;
                        } else if i == x.len() || y[j].0 < x[i].0 {
                            sum.push(y[j]);
                            j += 1;
                        } else {
                            let c = F::from_canonical_u64(x[i].1) + F::from_canonical_u64(y[j].1);
                            if !c.is_zero() {
                                sum.push((x[i].0, c.to_canonical_u64()));
                            }
                            i += 1;
                            j += 1;
                        }
                    }
                    (sum.len() <= WIDTH).then_some(sum)
                }
                _ => None,
            },
            Node::Mul(a, b) => match (constant(a), constant(b)) {
                (None, None) => Some(vec![(id as NodeId, 1)]),
                (Some(_), Some(_)) => None,
                (Some(c), None) | (None, Some(c)) => {
                    let x = if constant(a).is_some() { b } else { a };
                    let c = F::from_canonical_u64(c);
                    forms[x as usize].as_ref().map(|form| {
                        form.iter()
                            .map(|&(n, k)| (n, (F::from_canonical_u64(k) * c).to_canonical_u64()))
                            .filter(|&(_, k)| k != 0)
                            .collect()
                    })
                }
            },
            Node::Const(_) => None,
            _ => Some(vec![(id as NodeId, 1)]),
        };
        forms.push(form);
    }
    // Candidate sums grouped by their inputs.
    let mut groups: HashMap<Vec<NodeId>, Vec<NodeId>> = HashMap::new();
    for (id, node) in nodes.iter().enumerate() {
        if !matches!(node, Node::Add(..)) {
            continue;
        }
        let Some(form) = &forms[id] else { continue };
        if form.len() == WIDTH && form.iter().all(|&(_, c)| is_entry(c)) {
            let inputs = form.iter().map(|&(n, _)| n).collect::<Vec<_>>();
            groups.entry(inputs).or_default().push(id as NodeId);
        }
    }
    // The row of `form` under a column assignment, if it is one.
    let row_of = |form: &[(NodeId, u64)], column: &[usize]| {
        (0..WIDTH).find(|&r| (0..WIDTH).all(|i| form[i].1 == matrix[r][column[i]]))
    };
    // matched[sum] = (group, row); group inputs ordered by column.
    let mut matched: HashMap<NodeId, (usize, usize)> = HashMap::new();
    let mut group_inputs: Vec<[NodeId; WIDTH]> = Vec::new();
    let mut keys = groups.keys().cloned().collect::<Vec<_>>();
    keys.sort_unstable();
    for inputs in keys {
        let sums = &groups[&inputs];
        let first = forms[sums[0] as usize].as_ref().unwrap();
        // The assignment matching the most sums; a group may also hold sums
        // that are not rows (a row before its diagonal term is added).
        let mut best: Option<(Vec<usize>, Vec<Option<usize>>, usize)> = None;
        'rows: for r0 in 0..WIDTH {
            // Columns each input may take if the first sum is row r0, then
            // the bijections consistent with them, up to a bound.
            let options = (0..WIDTH)
                .map(|i| {
                    (0..WIDTH)
                        .filter(|&j| matrix[r0][j] == first[i].1)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            if options.iter().any(|o| o.is_empty()) {
                continue;
            }
            let mut column = vec![usize::MAX; WIDTH];
            let mut used = [false; WIDTH];
            let mut tried = 0;
            let all = bijections(&options, 0, &mut column, &mut used, &mut |column| {
                tried += 1;
                let rows = sums
                    .iter()
                    .map(|&s| row_of(forms[s as usize].as_ref().unwrap(), column))
                    .collect::<Vec<_>>();
                let count = rows.iter().flatten().count();
                if best.as_ref().is_none_or(|b| count > b.2) {
                    best = Some((column.to_vec(), rows, count));
                }
                if count == sums.len() {
                    Some(true)
                } else {
                    (tried >= 64).then_some(false)
                }
            });
            if all == Some(true) {
                break 'rows;
            }
        }
        let Some((column, rows, _)) = best else {
            continue;
        };
        let mut distinct = rows.iter().flatten().copied().collect::<Vec<_>>();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < MIN_MDS_ROWS {
            continue;
        }
        let mut ordered = [0; WIDTH];
        for (i, &j) in column.iter().enumerate() {
            ordered[j] = inputs[i];
        }
        let g = group_inputs.len();
        group_inputs.push(ordered);
        for (&s, &r) in sums.iter().zip(&rows) {
            if let Some(r) = r {
                matched.insert(s, (g, r));
            }
        }
    }
    if matched.is_empty() {
        return None;
    }
    // Rebuild; a new group is created at its first matched sum, after its
    // inputs, which precede every sum over them.
    let mut new = Dag::<F>::new();
    let mut map = vec![NodeId::MAX; nodes.len()];
    new.mds_groups = vec![[0; WIDTH]; dag.mds_groups.len()];
    let mut mds_done = vec![false; dag.mds_groups.len()];
    let mut outputs_of: Vec<Option<[NodeId; WIDTH]>> = vec![None; group_inputs.len()];
    let mut qmul_of = vec![[NodeId::MAX; 5]; dag.qmul_groups.len()];
    for (id, node) in nodes.iter().enumerate() {
        let m = |x: NodeId| map[x as usize];
        if let Some(&(g, r)) = matched.get(&(id as NodeId)) {
            if outputs_of[g].is_none() {
                outputs_of[g] = Some(new.mds(group_inputs[g].map(m)));
            }
            map[id] = outputs_of[g].unwrap()[r];
            continue;
        }
        map[id] = match *node {
            Node::Add(a, b) => new.push(Node::Add(m(a), m(b))),
            Node::Sub(a, b) => new.push(Node::Sub(m(a), m(b))),
            Node::Mul(a, b) => new.push(Node::Mul(m(a), m(b))),
            Node::Mds(group, k) => {
                if !mds_done[group as usize] {
                    mds_done[group as usize] = true;
                    new.mds_groups[group as usize] = dag.mds_groups[group as usize].map(m);
                }
                new.push(Node::Mds(group, k))
            }
            Node::Qmul(group, k) => {
                if qmul_of[group as usize][0] == NodeId::MAX {
                    let inputs = dag.qmul_groups[group as usize].map(m);
                    let mut x = [0; 5];
                    let mut y = [0; 5];
                    x.copy_from_slice(&inputs[..5]);
                    y.copy_from_slice(&inputs[5..]);
                    qmul_of[group as usize] = new.qmul(x, y);
                }
                qmul_of[group as usize][k as usize]
            }
            leaf => new.push(leaf),
        };
    }
    Some(GateTrace {
        constraints: trace.constraints.iter().map(|&c| map[c as usize]).collect(),
        dag: new,
    })
}

/// Rebuilds a trace with negations folded into the operations that use
/// them, preserving every constraint's value exactly.
///
/// Plonky2 builders subtract by multiplying by minus one (`sub(x, y)` is the
/// arithmetic `x + (-1) * y`), so traces contain many products by `p - 1` and
/// other constants just below the modulus. Each node is rebuilt as a value and
/// a sign: a product by a constant `c` with `-c` below 2^32 becomes a product
/// by `-c` (or no product for `c = -1`) with the sign flipped, products and
/// sums propagate signs, and an addition of a negated value becomes a
/// subtraction. A negated value is materialised as `0 - x` only where an
/// operation needs it positive: MDS and quintic inputs and the constraints.
pub(crate) fn negation_folded_trace<F: PrimeField64>(trace: &GateTrace<F>) -> GateTrace<F> {
    let dag = &trace.dag;
    let mut new = Dag::<F>::new();
    let zero = new.constant(F::ZERO);
    // The rebuilt value of each node: (node, negated).
    let mut map: Vec<(NodeId, bool)> = Vec::with_capacity(dag.nodes.len());
    let positive = |new: &mut Dag<F>, (n, negated): (NodeId, bool)| {
        if negated {
            new.sub(zero, n)
        } else {
            n
        }
    };
    // A constant whose negation is a small multiplier, as that negation.
    let negated_small = |new: &Dag<F>, n: NodeId| match new.nodes[n as usize] {
        Node::Const(c) if c >= 1 << 32 => {
            let neg = (-F::from_canonical_u64(c)).to_canonical_u64();
            (neg < 1 << 32).then_some(neg)
        }
        _ => None,
    };
    new.mds_groups = vec![[0; 12]; dag.mds_groups.len()];
    let mut mds_done = vec![false; dag.mds_groups.len()];
    let mut qmul_of = vec![[NodeId::MAX; 5]; dag.qmul_groups.len()];
    for node in &dag.nodes {
        let value = match *node {
            Node::Add(a, b) => {
                let ((x, sx), (y, sy)) = (map[a as usize], map[b as usize]);
                match (sx, sy) {
                    (false, false) => (new.add(x, y), false),
                    (false, true) => (new.sub(x, y), false),
                    (true, false) => (new.sub(y, x), false),
                    (true, true) => (new.add(x, y), true),
                }
            }
            Node::Sub(a, b) => {
                let ((x, sx), (y, sy)) = (map[a as usize], map[b as usize]);
                match (sx, sy) {
                    (false, false) => (new.sub(x, y), false),
                    (false, true) => (new.add(x, y), false),
                    (true, false) => (new.add(x, y), true),
                    (true, true) => (new.sub(y, x), false),
                }
            }
            Node::Mul(a, b) => {
                let ((mut x, sx), (mut y, sy)) = (map[a as usize], map[b as usize]);
                let mut negated = sx != sy;
                for operand in [&mut x, &mut y] {
                    if let Some(neg) = negated_small(&new, *operand) {
                        *operand = new.constant(F::from_canonical_u64(neg));
                        negated = !negated;
                    }
                }
                (new.mul(x, y), negated)
            }
            Node::Mds(group, k) => {
                if !mds_done[group as usize] {
                    mds_done[group as usize] = true;
                    let inputs = dag.mds_groups[group as usize].map(|i| map[i as usize]);
                    new.mds_groups[group as usize] = inputs.map(|v| positive(&mut new, v));
                }
                (new.push(Node::Mds(group, k)), false)
            }
            Node::Qmul(group, k) => {
                if qmul_of[group as usize][0] == NodeId::MAX {
                    let inputs = dag.qmul_groups[group as usize].map(|i| map[i as usize]);
                    let inputs = inputs.map(|v| positive(&mut new, v));
                    let mut x = [0; 5];
                    let mut y = [0; 5];
                    x.copy_from_slice(&inputs[..5]);
                    y.copy_from_slice(&inputs[5..]);
                    qmul_of[group as usize] = new.qmul(x, y);
                }
                (qmul_of[group as usize][k as usize], false)
            }
            leaf => (new.push(leaf), false),
        };
        map.push(value);
    }
    let constraints = trace
        .constraints
        .iter()
        .map(|&c| positive(&mut new, map[c as usize]))
        .collect();
    GateTrace {
        constraints,
        dag: new,
    }
}

/// WGSL for `fn <name>(row: u32, filter: Field, cs: ptr<function, array<Field, NC>>)`
/// adding `filter * constraint_k` to `cs[k]`. The body reads inputs through
/// `wire(row, i)`, `constant(row, j)` and `public_input(k)`, which the
/// enclosing module defines.
#[cfg(test)]
pub(crate) fn emit_gate<F: PrimeField64>(
    trace: &GateTrace<F>,
    name: &str,
    num_constraints: usize,
) -> String {
    let nodes = &trace.dag.nodes;
    let mut needed = vec![false; nodes.len()];
    let mut stack = trace.constraints.clone();
    while let Some(id) = stack.pop() {
        if needed[id as usize] {
            continue;
        }
        needed[id as usize] = true;
        match nodes[id as usize] {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                stack.push(a);
                stack.push(b);
            }
            Node::Mds(group, _) => stack.extend(trace.dag.mds_groups[group as usize]),
            Node::Qmul(group, _) => stack.extend(trace.dag.qmul_groups[group as usize]),
            _ => {}
        }
    }
    let mut emitted_groups = vec![false; trace.dag.mds_groups.len()];
    let mut emitted_qmuls = vec![false; trace.dag.qmul_groups.len()];
    let operand = |id: NodeId| match nodes[id as usize] {
        Node::Const(c) => literal(c),
        _ => format!("v{id}"),
    };
    let mut out = String::new();
    writeln!(
        out,
        "fn {name}(row: u32, gate_filter: Field, cs: ptr<function, array<Field, {num_constraints}>>) {{"
    )
    .unwrap();
    for (id, node) in nodes.iter().enumerate() {
        if !needed[id] {
            continue;
        }
        let expression = match *node {
            Node::Wire(i) => format!("wire(row, {i}u)"),
            Node::Constant(j) => format!("constant(row, {j}u)"),
            Node::PublicInput(k) => format!("public_input({k}u)"),
            Node::Const(_) => continue,
            Node::Add(a, b) => format!("add_field({}, {})", operand(a), operand(b)),
            Node::Sub(a, b) => format!("sub_field({}, {})", operand(a), operand(b)),
            Node::Mul(a, b) => format!("mul_field({}, {})", operand(a), operand(b)),
            Node::Mds(group, k) => {
                if !emitted_groups[group as usize] {
                    emitted_groups[group as usize] = true;
                    let inputs = trace.dag.mds_groups[group as usize]
                        .iter()
                        .map(|&input| operand(input))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(out, "    let g{group} = mds12(array<Field, 12>({inputs}));").unwrap();
                }
                format!("g{group}[{k}]")
            }
            Node::Qmul(group, k) => {
                if !emitted_qmuls[group as usize] {
                    emitted_qmuls[group as usize] = true;
                    let inputs = trace.dag.qmul_groups[group as usize];
                    let vector = |part: &[NodeId]| {
                        part.iter()
                            .map(|&input| operand(input))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    writeln!(
                        out,
                        "    let q{group} = qmul(array<Field, 5>({}), array<Field, 5>({}));",
                        vector(&inputs[..5]),
                        vector(&inputs[5..])
                    )
                    .unwrap();
                }
                format!("q{group}[{k}]")
            }
        };
        writeln!(out, "    let v{id} = {expression};").unwrap();
    }
    for (k, &constraint) in trace.constraints.iter().enumerate() {
        writeln!(
            out,
            "    (*cs)[{k}] = add_field((*cs)[{k}], mul_field(gate_filter, {}));",
            operand(constraint)
        )
        .unwrap();
    }
    out.push_str("}\n");
    out
}

/// Row `r` of the Poseidon MDS layer over `v`.
pub(crate) fn mds_row<F: RichField>(v: &[F; 12], r: usize) -> F {
    let mut acc = F::ZERO;
    for i in 0..SPONGE_WIDTH {
        acc +=
            v[(i + r) % SPONGE_WIDTH] * F::from_canonical_u64(<F as Poseidon>::MDS_MATRIX_CIRC[i]);
    }
    acc + v[r] * F::from_canonical_u64(<F as Poseidon>::MDS_MATRIX_DIAG[r])
}

/// Coefficient `k` of the product of the quintic extension elements `v[..5]`
/// and `v[5..]` in `F[X] / (X^5 - QMUL_W)`.
#[cfg(test)]
pub(crate) fn qmul_coeff<F: RichField>(v: &[F; 10], k: usize) -> F {
    let w = F::from_canonical_u64(QMUL_W);
    let mut acc = F::ZERO;
    for i in 0..5 {
        let j = (k + 5 - i) % 5;
        let term = v[i] * v[5 + j];
        acc += if i + j == k { term } else { term * w };
    }
    acc
}

/// Evaluates the trace on the host, for tests.
#[cfg(test)]
pub(crate) fn evaluate<F: RichField>(
    trace: &GateTrace<F>,
    wires: &[F],
    constants: &[F],
    public_inputs: &[F],
) -> Vec<F> {
    let mut values = Vec::with_capacity(trace.dag.nodes.len());
    for node in &trace.dag.nodes {
        let value = match *node {
            Node::Wire(i) => wires[i as usize],
            Node::Constant(j) => constants[j as usize],
            Node::PublicInput(k) => public_inputs[k as usize],
            Node::Const(c) => F::from_canonical_u64(c),
            Node::Add(a, b) => values[a as usize] + values[b as usize],
            Node::Sub(a, b) => values[a as usize] - values[b as usize],
            Node::Mul(a, b) => values[a as usize] * values[b as usize],
            Node::Mds(group, k) => {
                let inputs: [F; 12] =
                    trace.dag.mds_groups[group as usize].map(|i| values[i as usize]);
                mds_row(&inputs, k as usize)
            }
            Node::Qmul(group, k) => {
                let inputs: [F; 10] =
                    trace.dag.qmul_groups[group as usize].map(|i| values[i as usize]);
                qmul_coeff(&inputs, k as usize)
            }
        };
        values.push(value);
    }
    trace
        .constraints
        .iter()
        .map(|&id| values[id as usize])
        .collect()
}
