fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(stem) = args.next() else {
        anyhow::bail!("usage: witness-profile SNAPSHOT_STEM [ITERATIONS]");
    };
    if stem == "--help" || stem == "-h" {
        println!("usage: witness-profile SNAPSHOT_STEM [ITERATIONS]");
        return Ok(());
    }
    let iterations = args.next().map_or(Ok(24), |n| n.parse::<usize>())?;
    anyhow::ensure!((1..=10000).contains(&iterations), "iterations must be 1 to 10000");
    anyhow::ensure!(args.next().is_none(), "unexpected additional argument");
    plonky2::plonk::witness_program::replay_snapshot(std::path::Path::new(&stem), iterations)
}
