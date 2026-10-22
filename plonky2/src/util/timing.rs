use log::{log, Level};
#[cfg(feature = "timing")]
use web_time::{Duration, Instant};

/// Critical-path diagnostic: `PLONKY2_DELAY_SCOPE=<name>` and
/// `PLONKY2_DELAY_MS=<ms>` make the calling thread sleep at the start of
/// every timing scope with exactly that name. A delay that lengthens the
/// whole proof by its own length lies on the critical path; any shortfall is
/// the slack at that point (for example GPU work already in flight). Only in
/// `gpu-profiling` builds; the sleep is counted in the scope's own time.
#[cfg(all(feature = "timing", feature = "gpu-profiling", feature = "std"))]
fn injected_delay(ctx: &str) {
    static DELAY: std::sync::OnceLock<Option<(String, std::time::Duration)>> =
        std::sync::OnceLock::new();
    let delay = DELAY.get_or_init(|| {
        let scope = crate::util::switches::text("PLONKY2_DELAY_SCOPE")?;
        let ms = crate::util::switches::number::<f64>("PLONKY2_DELAY_MS")?;
        log::warn!("plonky2: injecting {ms} ms at the start of every \"{scope}\" scope");
        Some((scope, std::time::Duration::from_secs_f64(ms / 1000.0)))
    });
    if let Some((scope, duration)) = delay {
        if scope == ctx {
            std::thread::sleep(*duration);
        }
    }
}

/// The hierarchy of scopes, and the time consumed by each one. Useful for profiling.
#[cfg(feature = "timing")]
#[derive(Debug)]
pub struct TimingTree {
    /// The name of this scope.
    name: String,
    /// The level at which to log this scope and its children.
    level: log::Level,
    /// The time when this scope was created.
    enter_time: Instant,
    /// The time when this scope was destroyed, or None if it has not yet been destroyed.
    exit_time: Option<Instant>,
    /// Any child scopes.
    children: Vec<TimingTree>,
}

#[cfg(not(feature = "timing"))]
#[derive(Debug)]
pub struct TimingTree(Level);

#[cfg(feature = "timing")]
impl Default for TimingTree {
    fn default() -> Self {
        TimingTree::new("root", Level::Debug)
    }
}

#[cfg(not(feature = "timing"))]
impl Default for TimingTree {
    fn default() -> Self {
        TimingTree::new("", Level::Debug)
    }
}

impl TimingTree {
    #[cfg(feature = "timing")]
    pub fn new(root_name: &str, level: Level) -> Self {
        Self {
            name: root_name.to_string(),
            level,
            enter_time: Instant::now(),
            exit_time: None,
            children: vec![],
        }
    }

    #[cfg(not(feature = "timing"))]
    pub fn new(_root_name: &str, level: Level) -> Self {
        Self(level)
    }

    /// Whether this scope is still in scope.
    #[cfg(feature = "timing")]
    const fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// A description of the stack of currently-open scopes.
    #[cfg(feature = "timing")]
    pub fn open_stack(&self) -> String {
        let mut stack = Vec::new();
        self.open_stack_helper(&mut stack);
        stack.join(" > ")
    }

    #[cfg(feature = "timing")]
    fn open_stack_helper(&self, stack: &mut Vec<String>) {
        if self.is_open() {
            stack.push(self.name.clone());
            if let Some(last_child) = self.children.last() {
                last_child.open_stack_helper(stack);
            }
        }
    }

    #[cfg(feature = "timing")]
    pub fn push(&mut self, ctx: &str, mut level: log::Level) {
        assert!(self.is_open());

        // We don't want a scope's log level to be stronger than that of its parent.
        level = level.max(self.level);

        if let Some(last_child) = self.children.last_mut() {
            if last_child.is_open() {
                last_child.push(ctx, level);
                return;
            }
        }

        self.children.push(TimingTree {
            name: ctx.to_string(),
            level,
            enter_time: Instant::now(),
            exit_time: None,
            children: vec![],
        });
        #[cfg(all(feature = "gpu-profiling", feature = "std"))]
        injected_delay(ctx);
    }

    #[cfg(not(feature = "timing"))]
    pub fn push(&mut self, _ctx: &str, _level: log::Level) {}

    /// Close the deepest open scope from this tree.
    #[cfg(feature = "timing")]
    pub fn pop(&mut self) {
        assert!(self.is_open());

        if let Some(last_child) = self.children.last_mut() {
            if last_child.is_open() {
                last_child.pop();
                return;
            }
        }

        self.exit_time = Some(Instant::now());
    }

    #[cfg(not(feature = "timing"))]
    pub fn pop(&mut self) {}

    #[cfg(feature = "timing")]
    fn duration(&self) -> Duration {
        self.exit_time
            .unwrap_or_else(Instant::now)
            .duration_since(self.enter_time)
    }

    #[cfg(feature = "timing")]
    fn duration_at(&self, now: Instant) -> Duration {
        self.exit_time
            .unwrap_or(now)
            .duration_since(self.enter_time)
    }

    // Children are inclusive: subtract direct children only, never grandchildren.
    // A single snapshot also makes open scopes independent of logging latency.
    #[cfg(feature = "timing")]
    fn unattributed_at(&self, now: Instant) -> Duration {
        self.duration_at(now).saturating_sub(
            self.children
                .iter()
                .map(|child| child.duration_at(now))
                .sum(),
        )
    }

    /// Filter out children with a low duration.
    #[cfg(feature = "timing")]
    pub fn filter(&self, min_delta: Duration) -> Self {
        Self {
            name: self.name.clone(),
            level: self.level,
            enter_time: self.enter_time,
            exit_time: self.exit_time,
            children: self
                .children
                .iter()
                .filter(|c| c.duration() >= min_delta)
                .map(|c| c.filter(min_delta))
                .collect(),
        }
    }

    #[cfg(feature = "timing")]
    pub fn print(&self) {
        self.print_helper(0, Instant::now());
    }

    #[cfg(not(feature = "timing"))]
    pub fn print(&self) {
        log!(
            self.0,
            "TimingTree is not supported without the 'timing' feature enabled"
        );
    }

    #[cfg(feature = "timing")]
    fn print_helper(&self, depth: usize, now: Instant) {
        let prefix = "| ".repeat(depth);
        log!(
            self.level,
            "{}{:.4}s to {}",
            prefix,
            self.duration_at(now).as_secs_f64(),
            self.name
        );
        for child in &self.children {
            child.print_helper(depth + 1, now);
        }
        if !self.children.is_empty() {
            log!(
                self.level,
                "{}| {:.4}s to unattributed (parent minus direct children)",
                prefix,
                self.unattributed_at(now).as_secs_f64(),
            );
        }
    }
}

/// Creates a named scope; useful for debugging.
#[macro_export]
macro_rules! timed {
    ($timing_tree:expr, $level:expr, $ctx:expr, $exp:expr) => {{
        $timing_tree.push($ctx, $level);
        let res = $exp;
        $timing_tree.pop();
        res
    }};
    // If no context is specified, default to Debug.
    ($timing_tree:expr, $ctx:expr, $exp:expr) => {{
        $timing_tree.push($ctx, log::Level::Debug);
        let res = $exp;
        $timing_tree.pop();
        res
    }};
}

#[cfg(all(test, feature = "timing"))]
mod tests {
    use super::*;

    #[test]
    fn unattributed_subtracts_only_direct_children_at_one_snapshot() {
        let start = Instant::now();
        let scope = |name: &str, begin: u64, end: Option<u64>, children| TimingTree {
            name: name.into(),
            level: Level::Debug,
            enter_time: start + Duration::from_millis(begin),
            exit_time: end.map(|ms| start + Duration::from_millis(ms)),
            children,
        };
        let nested = scope("nested", 20, Some(40), vec![]);
        let child = scope("child", 10, Some(60), vec![nested]);
        let open = scope("open child", 80, None, vec![]);
        let root = scope("root", 0, None, vec![child, open]);
        let now = start + Duration::from_millis(100);
        assert_eq!(root.unattributed_at(now), Duration::from_millis(30));
        assert_eq!(
            root.children[0].unattributed_at(now),
            Duration::from_millis(30)
        );
    }
}
