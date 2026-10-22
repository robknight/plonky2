//! Diagnostic environment switches, read the same way everywhere. A flag is
//! on when the variable is set to anything but `0`. A value that does not
//! parse is ignored with one warning per variable. Settings that configure
//! the GPU context are read with the context, not here.
// Which helpers have callers depends on the enabled features.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::str::FromStr;
use std::string::String;
use std::sync::Mutex;

/// Whether `name` is set to anything but `0`.
pub(crate) fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| value != "0")
}

/// The value of `name` as text, if it is set and valid Unicode.
pub(crate) fn text(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(value)) => {
            warn_once(name, &value.to_string_lossy());
            None
        }
    }
}

/// The value of `name` parsed as `T`, if it is set and parses.
pub(crate) fn number<T: FromStr>(name: &str) -> Option<T> {
    let value = text(name)?;
    let parsed = value.parse().ok();
    if parsed.is_none() {
        warn_once(name, &value);
    }
    parsed
}

/// The path in `name`, if it is set and not empty.
pub(crate) fn path(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

fn warn_once(name: &str, value: &str) {
    static WARNED: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    if WARNED.lock().unwrap().insert(name.into()) {
        log::warn!("plonky2: ignoring {name}={value:?}, which does not parse");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test uses its own variable names, so tests running in parallel
    /// do not interfere.
    #[test]
    fn flags_numbers_and_paths() {
        let name = "PLONKY2_SWITCHES_TEST_FLAG";
        std::env::remove_var(name);
        assert!(!flag(name));
        for (value, on) in [("0", false), ("1", true), ("yes", true), ("", true)] {
            std::env::set_var(name, value);
            assert_eq!(flag(name), on, "{value:?}");
        }
        let name = "PLONKY2_SWITCHES_TEST_NUMBER";
        std::env::set_var(name, "12");
        assert_eq!(number::<u32>(name), Some(12));
        std::env::set_var(name, "twelve");
        assert_eq!(number::<u32>(name), None);
        let name = "PLONKY2_SWITCHES_TEST_PATH";
        std::env::set_var(name, "");
        assert_eq!(path(name), None);
        std::env::set_var(name, "out");
        assert_eq!(path(name), Some("out".into()));
    }
}
