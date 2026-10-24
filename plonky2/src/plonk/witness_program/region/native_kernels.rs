use super::*;
use crate::field::goldilocks_field::GoldilocksField as G;
use crate::field::types::Field;
#[allow(unused_parens)]
fn p0(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let m0 = i0 * i1;
    let z0 = i2;
    let v0 = (m0 + z0);
    e.output(o, 3, count, k, v0)?;
    let m1 = v0 * i1;
    let z1 = i2;
    let v1 = (m1 + z1);
    e.output(o, 4, count, k, v1)?;
    let m2 = v1 * i1;
    let z2 = i2;
    let v2 = (m2 + z2);
    e.output(o, 5, count, k, v2)?;
    let m3 = v2 * i1;
    let z3 = i2;
    let v3 = (m3 + z3);
    e.output(o, 6, count, k, v3)?;
    let m4 = v3 * i1;
    let z4 = i2;
    let v4 = (m4 + z4);
    e.output(o, 7, count, k, v4)?;
    let m5 = v4 * i1;
    let z5 = i2;
    let v5 = (m5 + z5);
    e.output(o, 8, count, k, v5)?;
    let m6 = v5 * i1;
    let z6 = i2;
    let v6 = (m6 + z6);
    e.output(o, 9, count, k, v6)?;
    let m7 = v6 * i1;
    let z7 = i2;
    let v7 = (m7 + z7);
    e.output(o, 10, count, k, v7)?;
    Ok(())
}
#[allow(unused_parens)]
fn p1(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let m0 = i0 * i1;
    let z0 = i2;
    let v0 = (m0 + z0);
    e.output(o, 3, count, k, v0)?;
    let m1 = v0 * i1;
    let z1 = i2;
    let v1 = (m1 + z1);
    e.output(o, 4, count, k, v1)?;
    let m2 = v1 * i1;
    let z2 = i2;
    let v2 = (m2 + z2);
    e.output(o, 5, count, k, v2)?;
    let m3 = v2 * i1;
    let z3 = i2;
    let v3 = (m3 + z3);
    e.output(o, 6, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p2(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let m0 = i0 * i1;
    let z0 = i1;
    let v0 = (m0 + -(z0));
    e.output(o, 3, count, k, v0)?;
    let m1 = i0 * i2;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 4, count, k, v1)?;
    Ok(())
}
#[allow(unused_parens)]
fn p3(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let m0 = i0 * i0;
    let z0 = i1;
    let v0 = (m0 + -(z0));
    e.output(o, 3, count, k, v0)?;
    let m1 = i2 * i0;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 4, count, k, v1)?;
    Ok(())
}
#[allow(unused_parens)]
fn p4(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 4, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = i3;
    let v1 = (m1 + -(z1));
    e.output(o, 5, count, k, v1)?;
    Ok(())
}
#[allow(unused_parens)]
fn p5(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 5, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 6, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 7, count, k, v2)?;
    let m3 = v2 * i4;
    let v3 = m3;
    e.output(o, 8, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p6(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 4, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 5, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 6, count, k, v2)?;
    Ok(())
}
#[allow(unused_parens)]
fn p7(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 5, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 6, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 7, count, k, v2)?;
    let m3 = i4 * v2;
    let v3 = m3;
    e.output(o, 8, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p8(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let i6 = e.input(o, 6, count, k)?;
    let i7 = e.input(o, 7, count, k)?;
    let i8 = e.input(o, 8, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 9, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 10, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 11, count, k, v2)?;
    let m3 = v2 * i4;
    let v3 = m3;
    e.output(o, 12, count, k, v3)?;
    let m4 = v3 * i5;
    let v4 = m4;
    e.output(o, 13, count, k, v4)?;
    let m5 = v4 * i6;
    let v5 = m5;
    e.output(o, 14, count, k, v5)?;
    let m6 = v5 * i7;
    let v6 = m6;
    e.output(o, 15, count, k, v6)?;
    let m7 = v6 * i8;
    let v7 = m7;
    e.output(o, 16, count, k, v7)?;
    Ok(())
}
#[allow(unused_parens)]
fn p9(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let z0 = i2;
    let v0 = (m0 + -(z0));
    e.output(o, 6, count, k, v0)?;
    let m1 = i3 * v0;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 7, count, k, v1)?;
    let m2 = i3 * i4;
    let z2 = v1;
    let v2 = (m2 + -(z2));
    e.output(o, 8, count, k, v2)?;
    let m3 = i5 * v2;
    let z3 = v2;
    let v3 = (m3 + -(z3));
    e.output(o, 9, count, k, v3)?;
    let m4 = i5 * i4;
    let z4 = v3;
    let v4 = (m4 + -(z4));
    e.output(o, 10, count, k, v4)?;
    Ok(())
}
#[allow(unused_parens)]
fn p10(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let m0 = i0 * i0;
    let z0 = i1;
    let v0 = (m0 + z0);
    e.output(o, 2, count, k, v0)?;
    let m1 = v0 * i0;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 3, count, k, v1)?;
    let m2 = v1 * i0;
    let z2 = i1;
    let v2 = (m2 + z2);
    e.output(o, 4, count, k, v2)?;
    let m3 = v2 * i0;
    let z3 = i1;
    let v3 = (m3 + z3);
    e.output(o, 5, count, k, v3)?;
    let m4 = v3 * i0;
    let z4 = i1;
    let v4 = (m4 + z4);
    e.output(o, 6, count, k, v4)?;
    let m5 = v4 * i0;
    let z5 = i1;
    let v5 = (m5 + z5);
    e.output(o, 7, count, k, v5)?;
    let m6 = v5 * i0;
    let z6 = i1;
    let v6 = (m6 + z6);
    e.output(o, 8, count, k, v6)?;
    let m7 = v6 * i0;
    let z7 = i1;
    let v7 = (m7 + z7);
    e.output(o, 9, count, k, v7)?;
    Ok(())
}
#[allow(unused_parens)]
fn p11(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let m0 = i0 * i1;
    let z0 = i2;
    let v0 = (m0 + -(z0));
    e.output(o, 4, count, k, v0)?;
    let m1 = i3 * v0;
    let v1 = m1;
    e.output(o, 5, count, k, v1)?;
    Ok(())
}
#[allow(unused_parens)]
fn p12(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let z0 = i2;
    let v0 = (m0 + -(z0));
    e.output(o, 6, count, k, v0)?;
    let m1 = i3 * v0;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 7, count, k, v1)?;
    let m2 = i3 * i4;
    let z2 = v1;
    let v2 = (m2 + -(z2));
    e.output(o, 8, count, k, v2)?;
    let m3 = i5 * v2;
    let z3 = v2;
    let v3 = (m3 + -(z3));
    e.output(o, 9, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p13(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 3, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 4, count, k, v1)?;
    Ok(())
}
#[allow(unused_parens)]
fn p14(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let m0 = i0 * i0;
    let z0 = i1;
    let v0 = (m0 + z0);
    e.output(o, 2, count, k, v0)?;
    let m1 = v0 * i0;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 3, count, k, v1)?;
    let m2 = v1 * i0;
    let z2 = i1;
    let v2 = (m2 + z2);
    e.output(o, 4, count, k, v2)?;
    let m3 = v2 * i0;
    let z3 = i1;
    let v3 = (m3 + z3);
    e.output(o, 5, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p15(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 6, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 7, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 8, count, k, v2)?;
    let m3 = v2 * i4;
    let v3 = m3;
    e.output(o, 9, count, k, v3)?;
    let m4 = i5 * v3;
    let v4 = m4;
    e.output(o, 10, count, k, v4)?;
    Ok(())
}
#[allow(unused_parens)]
fn p16(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 6, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 7, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 8, count, k, v2)?;
    let m3 = i4 * i4;
    let z3 = v2;
    let v3 = (m3 + -(z3));
    e.output(o, 9, count, k, v3)?;
    let m4 = v3 * i5;
    let v4 = m4;
    e.output(o, 10, count, k, v4)?;
    Ok(())
}
#[allow(unused_parens)]
fn p17(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let i6 = e.input(o, 6, count, k)?;
    let i7 = e.input(o, 7, count, k)?;
    let i8 = e.input(o, 8, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 9, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 10, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 11, count, k, v2)?;
    let m3 = v2 * i4;
    let v3 = m3;
    e.output(o, 12, count, k, v3)?;
    let m4 = v3 * i5;
    let v4 = m4;
    e.output(o, 13, count, k, v4)?;
    let m5 = v4 * i6;
    let v5 = m5;
    e.output(o, 14, count, k, v5)?;
    let m6 = v5 * i7;
    let v6 = m6;
    e.output(o, 15, count, k, v6)?;
    let m7 = i8 * v6;
    let v7 = m7;
    e.output(o, 16, count, k, v7)?;
    Ok(())
}
#[allow(unused_parens)]
fn p18(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 5, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 6, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 7, count, k, v2)?;
    let m3 = i4 * i4;
    let z3 = v2;
    let v3 = (m3 + -(z3));
    e.output(o, 8, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p19(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let m0 = i0 * i1;
    let z0 = i0;
    let v0 = (-(m0) + z0);
    e.output(o, 4, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 5, count, k, v1)?;
    let m2 = v1 * i3;
    let z2 = v1;
    let v2 = (-(m2) + z2);
    e.output(o, 6, count, k, v2)?;
    let m3 = v2 * i2;
    let z3 = i3;
    let v3 = (m3 + z3);
    e.output(o, 7, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p20(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let m0 = i0 * i1;
    let z0 = i0;
    let v0 = (-(m0) + z0);
    e.output(o, 5, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 6, count, k, v1)?;
    let m2 = v1 * i3;
    let z2 = i4;
    let v2 = (m2 + -(z2));
    e.output(o, 7, count, k, v2)?;
    Ok(())
}
#[allow(unused_parens)]
fn p21(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let z0 = i0;
    let v0 = (-(m0) + z0);
    e.output(o, 6, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 7, count, k, v1)?;
    let m2 = v1 * i3;
    let z2 = v1;
    let v2 = (-(m2) + z2);
    e.output(o, 8, count, k, v2)?;
    let m3 = v2 * i2;
    let z3 = i3;
    let v3 = (m3 + z3);
    e.output(o, 9, count, k, v3)?;
    let m4 = v3 * i4;
    let z4 = v3;
    let v4 = (-(m4) + z4);
    e.output(o, 10, count, k, v4)?;
    let m5 = v4 * i2;
    let z5 = i4;
    let v5 = (m5 + z5);
    e.output(o, 11, count, k, v5)?;
    let m6 = v5 * i5;
    let z6 = v5;
    let v6 = (-(m6) + z6);
    e.output(o, 12, count, k, v6)?;
    let m7 = v6 * i2;
    let z7 = i5;
    let v7 = (m7 + z7);
    e.output(o, 13, count, k, v7)?;
    Ok(())
}
#[allow(unused_parens)]
fn p22(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let i6 = e.input(o, 6, count, k)?;
    let i7 = e.input(o, 7, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 8, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 9, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 10, count, k, v2)?;
    let m3 = i4 * v2;
    let v3 = m3;
    e.output(o, 11, count, k, v3)?;
    let m4 = v3 * i5;
    let z4 = v3;
    let v4 = (-(m4) + z4);
    e.output(o, 12, count, k, v4)?;
    let m5 = v4 * i6;
    let z5 = i5;
    let v5 = (m5 + z5);
    e.output(o, 13, count, k, v5)?;
    let m6 = v5 * i7;
    let v6 = m6;
    e.output(o, 14, count, k, v6)?;
    Ok(())
}
#[allow(unused_parens)]
fn p23(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let m0 = i0 * i1;
    let z0 = i0;
    let v0 = (-(m0) + z0);
    e.output(o, 4, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 5, count, k, v1)?;
    let m2 = i3 * v1;
    let v2 = m2;
    e.output(o, 6, count, k, v2)?;
    Ok(())
}
#[allow(unused_parens)]
fn p24(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let m0 = i0 * i1;
    let z0 = i1;
    let v0 = (m0 + -(z0));
    e.output(o, 5, count, k, v0)?;
    let m1 = i0 * i2;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 6, count, k, v1)?;
    let m2 = i3 * v1;
    let z2 = i4;
    let v2 = (m2 + -(z2));
    e.output(o, 7, count, k, v2)?;
    Ok(())
}
#[allow(unused_parens)]
fn p25(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let m0 = i0 * i0;
    let v0 = m0;
    e.output(o, 1, count, k, v0)?;
    let m1 = v0 * v0;
    let v1 = m1;
    e.output(o, 2, count, k, v1)?;
    let m2 = v1 * v1;
    let v2 = m2;
    e.output(o, 3, count, k, v2)?;
    let m3 = v2 * v2;
    let v3 = m3;
    e.output(o, 4, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p26(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let m0 = i0 * i1;
    let z0 = i2;
    let v0 = (m0 + z0);
    e.output(o, 5, count, k, v0)?;
    let m1 = i0 * v0;
    let z1 = i3;
    let v1 = (m1 + z1);
    e.output(o, 6, count, k, v1)?;
    let m2 = i0 * v1;
    let z2 = i4;
    let v2 = (m2 + z2);
    e.output(o, 7, count, k, v2)?;
    Ok(())
}
#[allow(unused_parens)]
fn p27(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let z0 = i1;
    let v0 = (m0 + -(z0));
    e.output(o, 6, count, k, v0)?;
    let m1 = i0 * i2;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 7, count, k, v1)?;
    let m2 = i3 * v1;
    let z2 = i4;
    let v2 = (m2 + -(z2));
    e.output(o, 8, count, k, v2)?;
    let m3 = v2 * i5;
    let z3 = i5;
    let v3 = (m3 + -(z3));
    e.output(o, 9, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p28(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let i6 = e.input(o, 6, count, k)?;
    let m0 = i0 * i1;
    let z0 = i1;
    let v0 = (m0 + -(z0));
    e.output(o, 7, count, k, v0)?;
    let m1 = i0 * i2;
    let z1 = v0;
    let v1 = (m1 + -(z1));
    e.output(o, 8, count, k, v1)?;
    let m2 = i3 * v1;
    let z2 = i4;
    let v2 = (m2 + -(z2));
    e.output(o, 9, count, k, v2)?;
    let m3 = i5 * i6;
    let z3 = v2;
    let v3 = (m3 + -(z3));
    e.output(o, 10, count, k, v3)?;
    Ok(())
}
#[allow(unused_parens)]
fn p29(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let i6 = e.input(o, 6, count, k)?;
    let i7 = e.input(o, 7, count, k)?;
    let m0 = i0 * i1;
    let v0 = m0;
    e.output(o, 8, count, k, v0)?;
    let m1 = v0 * i2;
    let v1 = m1;
    e.output(o, 9, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 10, count, k, v2)?;
    let m3 = v2 * i4;
    let v3 = m3;
    e.output(o, 11, count, k, v3)?;
    let m4 = v3 * i5;
    let v4 = m4;
    e.output(o, 12, count, k, v4)?;
    let m5 = v4 * i6;
    let v5 = m5;
    e.output(o, 13, count, k, v5)?;
    let m6 = v5 * i7;
    let v6 = m6;
    e.output(o, 14, count, k, v6)?;
    Ok(())
}
#[allow(unused_parens)]
fn p30(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let m0 = i0 * i1;
    let z0 = i0;
    let v0 = (-(m0) + z0);
    e.output(o, 4, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = i1;
    let v1 = (m1 + z1);
    e.output(o, 5, count, k, v1)?;
    let m2 = v1 * i3;
    let v2 = m2;
    e.output(o, 6, count, k, v2)?;
    Ok(())
}
#[allow(unused_parens)]
fn p31(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    let o = g.targets_offset;
    let count = g.count;
    let i0 = e.input(o, 0, count, k)?;
    let i1 = e.input(o, 1, count, k)?;
    let i2 = e.input(o, 2, count, k)?;
    let i3 = e.input(o, 3, count, k)?;
    let i4 = e.input(o, 4, count, k)?;
    let i5 = e.input(o, 5, count, k)?;
    let m0 = i0 * i1;
    let z0 = i0;
    let v0 = ((m0 * G::from_noncanonical_u64(18442240469788262400u64)) + z0);
    e.output(o, 6, count, k, v0)?;
    let m1 = v0 * i2;
    let z1 = v0;
    let v1 = ((m1 * G::from_noncanonical_u64(18446742969902956800u64)) + z1);
    e.output(o, 7, count, k, v1)?;
    let m2 = v1 * i3;
    let z2 = v1;
    let v2 = ((m2 * G::from_noncanonical_u64(18446462594437873664u64)) + z2);
    e.output(o, 8, count, k, v2)?;
    let m3 = v2 * i4;
    let z3 = v2;
    let v3 = ((m3 * G::from_noncanonical_u64(18446744069414584319u64)) + z3);
    e.output(o, 9, count, k, v3)?;
    let m4 = v3 * i5;
    let v4 = m4;
    e.output(o, 10, count, k, v4)?;
    Ok(())
}
static KERNELS: &[PatternFn] = &[
    pattern_fallback,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
    p19,
    p20,
    p21,
    p22,
    p23,
    p24,
    p25,
    p26,
    p27,
    p28,
    p29,
    p30,
    p31,
];
static DESCRIPTORS: &[&[u64]] = &[
    &[
        3,
        8,
        0,
        1,
        2,
        1,
        1,
        9223372036854775808,
        1,
        2,
        1,
        1,
        9223372036854775809,
        1,
        2,
        1,
        1,
        9223372036854775810,
        1,
        2,
        1,
        1,
        9223372036854775811,
        1,
        2,
        1,
        1,
        9223372036854775812,
        1,
        2,
        1,
        1,
        9223372036854775813,
        1,
        2,
        1,
        1,
        9223372036854775814,
        1,
        2,
        1,
        1,
    ],
    &[
        3,
        4,
        0,
        1,
        2,
        1,
        1,
        9223372036854775808,
        1,
        2,
        1,
        1,
        9223372036854775809,
        1,
        2,
        1,
        1,
        9223372036854775810,
        1,
        2,
        1,
        1,
    ],
    &[
        3,
        2,
        0,
        1,
        1,
        1,
        18446744069414584320,
        0,
        2,
        9223372036854775808,
        1,
        18446744069414584320,
    ],
    &[
        3,
        2,
        0,
        0,
        1,
        1,
        18446744069414584320,
        2,
        0,
        9223372036854775808,
        1,
        18446744069414584320,
    ],
    &[
        4,
        2,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        3,
        1,
        18446744069414584320,
    ],
    &[
        5,
        4,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        9223372036854775810,
        4,
        9223372036854775810,
        1,
        0,
    ],
    &[
        4,
        3,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
    ],
    &[
        5,
        4,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        4,
        9223372036854775810,
        4,
        1,
        0,
    ],
    &[
        9,
        8,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        9223372036854775810,
        4,
        9223372036854775810,
        1,
        0,
        9223372036854775811,
        5,
        9223372036854775811,
        1,
        0,
        9223372036854775812,
        6,
        9223372036854775812,
        1,
        0,
        9223372036854775813,
        7,
        9223372036854775813,
        1,
        0,
        9223372036854775814,
        8,
        9223372036854775814,
        1,
        0,
    ],
    &[
        6,
        5,
        0,
        1,
        2,
        1,
        18446744069414584320,
        3,
        9223372036854775808,
        9223372036854775808,
        1,
        18446744069414584320,
        3,
        4,
        9223372036854775809,
        1,
        18446744069414584320,
        5,
        9223372036854775810,
        9223372036854775810,
        1,
        18446744069414584320,
        5,
        4,
        9223372036854775811,
        1,
        18446744069414584320,
    ],
    &[
        2,
        8,
        0,
        0,
        1,
        1,
        1,
        9223372036854775808,
        0,
        1,
        1,
        1,
        9223372036854775809,
        0,
        1,
        1,
        1,
        9223372036854775810,
        0,
        1,
        1,
        1,
        9223372036854775811,
        0,
        1,
        1,
        1,
        9223372036854775812,
        0,
        1,
        1,
        1,
        9223372036854775813,
        0,
        1,
        1,
        1,
        9223372036854775814,
        0,
        1,
        1,
        1,
    ],
    &[
        4,
        2,
        0,
        1,
        2,
        1,
        18446744069414584320,
        3,
        9223372036854775808,
        3,
        1,
        0,
    ],
    &[
        6,
        4,
        0,
        1,
        2,
        1,
        18446744069414584320,
        3,
        9223372036854775808,
        9223372036854775808,
        1,
        18446744069414584320,
        3,
        4,
        9223372036854775809,
        1,
        18446744069414584320,
        5,
        9223372036854775810,
        9223372036854775810,
        1,
        18446744069414584320,
    ],
    &[
        3,
        2,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
    ],
    &[
        2,
        4,
        0,
        0,
        1,
        1,
        1,
        9223372036854775808,
        0,
        1,
        1,
        1,
        9223372036854775809,
        0,
        1,
        1,
        1,
        9223372036854775810,
        0,
        1,
        1,
        1,
    ],
    &[
        6,
        5,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        9223372036854775810,
        4,
        9223372036854775810,
        1,
        0,
        5,
        9223372036854775811,
        5,
        1,
        0,
    ],
    &[
        6,
        5,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        4,
        4,
        9223372036854775810,
        1,
        18446744069414584320,
        9223372036854775811,
        5,
        9223372036854775811,
        1,
        0,
    ],
    &[
        9,
        8,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        9223372036854775810,
        4,
        9223372036854775810,
        1,
        0,
        9223372036854775811,
        5,
        9223372036854775811,
        1,
        0,
        9223372036854775812,
        6,
        9223372036854775812,
        1,
        0,
        9223372036854775813,
        7,
        9223372036854775813,
        1,
        0,
        8,
        9223372036854775814,
        8,
        1,
        0,
    ],
    &[
        5,
        4,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        4,
        4,
        9223372036854775810,
        1,
        18446744069414584320,
    ],
    &[
        4,
        4,
        0,
        1,
        0,
        18446744069414584320,
        1,
        9223372036854775808,
        2,
        1,
        1,
        1,
        9223372036854775809,
        3,
        9223372036854775809,
        18446744069414584320,
        1,
        9223372036854775810,
        2,
        3,
        1,
        1,
    ],
    &[
        5,
        3,
        0,
        1,
        0,
        18446744069414584320,
        1,
        9223372036854775808,
        2,
        1,
        1,
        1,
        9223372036854775809,
        3,
        4,
        1,
        18446744069414584320,
    ],
    &[
        6,
        8,
        0,
        1,
        0,
        18446744069414584320,
        1,
        9223372036854775808,
        2,
        1,
        1,
        1,
        9223372036854775809,
        3,
        9223372036854775809,
        18446744069414584320,
        1,
        9223372036854775810,
        2,
        3,
        1,
        1,
        9223372036854775811,
        4,
        9223372036854775811,
        18446744069414584320,
        1,
        9223372036854775812,
        2,
        4,
        1,
        1,
        9223372036854775813,
        5,
        9223372036854775813,
        18446744069414584320,
        1,
        9223372036854775814,
        2,
        5,
        1,
        1,
    ],
    &[
        8,
        7,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        4,
        9223372036854775810,
        4,
        1,
        0,
        9223372036854775811,
        5,
        9223372036854775811,
        18446744069414584320,
        1,
        9223372036854775812,
        6,
        5,
        1,
        1,
        9223372036854775813,
        7,
        9223372036854775813,
        1,
        0,
    ],
    &[
        4,
        3,
        0,
        1,
        0,
        18446744069414584320,
        1,
        9223372036854775808,
        2,
        1,
        1,
        1,
        3,
        9223372036854775809,
        3,
        1,
        0,
    ],
    &[
        5,
        3,
        0,
        1,
        1,
        1,
        18446744069414584320,
        0,
        2,
        9223372036854775808,
        1,
        18446744069414584320,
        3,
        9223372036854775809,
        4,
        1,
        18446744069414584320,
    ],
    &[
        1,
        4,
        0,
        0,
        0,
        1,
        0,
        9223372036854775808,
        9223372036854775808,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        9223372036854775809,
        9223372036854775809,
        1,
        0,
        9223372036854775810,
        9223372036854775810,
        9223372036854775810,
        1,
        0,
    ],
    &[
        5,
        3,
        0,
        1,
        2,
        1,
        1,
        0,
        9223372036854775808,
        3,
        1,
        1,
        0,
        9223372036854775809,
        4,
        1,
        1,
    ],
    &[
        6,
        4,
        0,
        1,
        1,
        1,
        18446744069414584320,
        0,
        2,
        9223372036854775808,
        1,
        18446744069414584320,
        3,
        9223372036854775809,
        4,
        1,
        18446744069414584320,
        9223372036854775810,
        5,
        5,
        1,
        18446744069414584320,
    ],
    &[
        7,
        4,
        0,
        1,
        1,
        1,
        18446744069414584320,
        0,
        2,
        9223372036854775808,
        1,
        18446744069414584320,
        3,
        9223372036854775809,
        4,
        1,
        18446744069414584320,
        5,
        6,
        9223372036854775810,
        1,
        18446744069414584320,
    ],
    &[
        8,
        7,
        0,
        1,
        0,
        1,
        0,
        9223372036854775808,
        2,
        9223372036854775808,
        1,
        0,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
        9223372036854775810,
        4,
        9223372036854775810,
        1,
        0,
        9223372036854775811,
        5,
        9223372036854775811,
        1,
        0,
        9223372036854775812,
        6,
        9223372036854775812,
        1,
        0,
        9223372036854775813,
        7,
        9223372036854775813,
        1,
        0,
    ],
    &[
        4,
        3,
        0,
        1,
        0,
        18446744069414584320,
        1,
        9223372036854775808,
        2,
        1,
        1,
        1,
        9223372036854775809,
        3,
        9223372036854775809,
        1,
        0,
    ],
    &[
        6,
        5,
        0,
        1,
        0,
        18442240469788262400,
        1,
        9223372036854775808,
        2,
        9223372036854775808,
        18446742969902956800,
        1,
        9223372036854775809,
        3,
        9223372036854775809,
        18446462594437873664,
        1,
        9223372036854775810,
        4,
        9223372036854775810,
        18446744069414584319,
        1,
        9223372036854775811,
        5,
        9223372036854775811,
        1,
        0,
    ],
];
pub(super) fn run(e: &Executor<'_>) -> Result<()> {
    run_adaptive(e, DESCRIPTORS, KERNELS)
}
