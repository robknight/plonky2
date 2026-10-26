//! Strict extraction of runtime configuration from process environment variables.
//!
//! GPU and witness configuration expose typed `from_env` methods. Unlike the
//! diagnostic switches, invalid runtime settings are errors rather than ignored
//! values. Paths retain their native operating-system representation.

use std::ffi::OsString;
use std::path::PathBuf;
use std::str::FromStr;

/// A runtime setting could not be decoded into its documented type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentError {
    /// The variable actually selected, including a migration alias if used.
    pub variable: &'static str,
    /// The accepted type or values.
    pub expected: String,
}
impl core::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: expected {}", self.variable, self.expected)
    }
}
impl std::error::Error for EnvironmentError {}

pub(crate) struct Environment<F = fn(&str) -> Option<OsString>> {
    lookup: F,
}
impl Environment {
    pub(crate) fn process() -> Self {
        Self {
            lookup: |key| std::env::var_os(key),
        }
    }
}
impl<F: Fn(&str) -> Option<OsString>> Environment<F> {
    #[cfg(test)]
    pub(crate) fn with_lookup(lookup: F) -> Self {
        Self { lookup }
    }
    fn value(&self, name: &'static str, alias: &'static str) -> Option<(&'static str, OsString)> {
        (self.lookup)(name).map(|v| (name, v)).or_else(|| {
            (name != alias)
                .then(|| (self.lookup)(alias))
                .flatten()
                .map(|v| (alias, v))
        })
    }
    pub(crate) fn present(&self, name: &str) -> bool {
        (self.lookup)(name).is_some()
    }
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn text(
        &self,
        name: &'static str,
        alias: &'static str,
    ) -> Result<Option<String>, EnvironmentError> {
        self.value(name, alias)
            .map(|(variable, value)| {
                value.into_string().map_err(|_| EnvironmentError {
                    variable,
                    expected: "Unicode text".into(),
                })
            })
            .transpose()
    }
    pub(crate) fn number<T: FromStr>(
        &self,
        name: &'static str,
        alias: &'static str,
    ) -> Result<Option<T>, EnvironmentError> {
        let Some((variable, value)) = self.value(name, alias) else {
            return Ok(None);
        };
        let text = value.into_string().map_err(|_| EnvironmentError {
            variable,
            expected: "Unicode numeric text".into(),
        })?;
        text.parse().map(Some).map_err(|_| EnvironmentError {
            variable,
            expected: format!("a valid {} number", core::any::type_name::<T>()),
        })
    }
    /// Outer None means absent; Some(None) restores automatic sizing.
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn optional_number<T: FromStr>(
        &self,
        name: &'static str,
        alias: &'static str,
    ) -> Result<Option<Option<T>>, EnvironmentError> {
        let Some((variable, value)) = self.value(name, alias) else {
            return Ok(None);
        };
        if value == "auto" {
            return Ok(Some(None));
        }
        value
            .to_str()
            .and_then(|s| s.parse().ok())
            .map(|v| Some(Some(v)))
            .ok_or_else(|| EnvironmentError {
                variable,
                expected: format!("auto or a valid {} number", core::any::type_name::<T>()),
            })
    }
    pub(crate) fn choice<T: Copy>(
        &self,
        name: &'static str,
        alias: &'static str,
        choices: &[(&str, T)],
    ) -> Result<Option<T>, EnvironmentError> {
        let Some((variable, value)) = self.value(name, alias) else {
            return Ok(None);
        };
        let parsed = value.to_str().and_then(|s| {
            choices
                .iter()
                .find(|(key, _)| *key == s)
                .map(|(_, value)| *value)
        });
        parsed.map(Some).ok_or_else(|| EnvironmentError {
            variable,
            expected: choices
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>()
                .join(", "),
        })
    }
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn boolean(
        &self,
        name: &'static str,
        alias: &'static str,
    ) -> Result<Option<bool>, EnvironmentError> {
        self.choice(
            name,
            alias,
            &[("true", true), ("1", true), ("false", false), ("0", false)],
        )
    }
    /// An absent or empty path disables storage; non-Unicode paths are valid.
    pub(crate) fn path(&self, name: &str) -> Option<PathBuf> {
        (self.lookup)(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_types_alias_precedence_and_native_paths() {
        let values: std::collections::BTreeMap<_, OsString> = [
            ("PRIMARY", "2".into()),
            ("OLD", "bad".into()),
            ("BOOL", "false".into()),
            ("EMPTY", "".into()),
            ("AUTO", "auto".into()),
            ("OVERFLOW", "18446744073709551616".into()),
        ]
        .into();
        let env = Environment::with_lookup(|key: &str| values.get(key).cloned());
        assert_eq!(env.number::<usize>("PRIMARY", "OLD").unwrap(), Some(2));
        assert_eq!(
            env.number::<usize>("UNSET", "OLD").unwrap_err().variable,
            "OLD"
        );
        assert_eq!(env.boolean("BOOL", "BOOL").unwrap(), Some(false));
        assert!(env.boolean("PRIMARY", "PRIMARY").is_err());
        assert!(env.number::<u64>("OVERFLOW", "OVERFLOW").is_err());
        assert!(env.number::<u32>("EMPTY", "EMPTY").is_err());
        assert_eq!(env.path("EMPTY"), None);
        assert_eq!(
            env.optional_number::<u64>("PRIMARY", "OLD").unwrap(),
            Some(Some(2))
        );
        assert_eq!(
            env.optional_number::<u64>("AUTO", "AUTO").unwrap(),
            Some(None)
        );
        assert_eq!(env.optional_number::<u64>("UNSET", "UNSET").unwrap(), None);
        assert!(env.optional_number::<u64>("OVERFLOW", "OVERFLOW").is_err());
        assert_eq!(env.number::<u64>("UNSET", "UNSET").unwrap(), None);
    }
    #[test]
    #[cfg(unix)]
    fn non_unicode_is_valid_for_paths_but_not_typed_text() {
        use std::os::unix::ffi::OsStringExt;
        let value = OsString::from_vec(vec![b'/', 0xff]);
        let env = Environment::with_lookup(|_: &str| Some(value.clone()));
        assert_eq!(env.path("PATH"), Some(PathBuf::from(&value)));
        assert_eq!(env.text("TEXT", "OLD").unwrap_err().variable, "TEXT");
        assert!(env.number::<u64>("NUMBER", "NUMBER").is_err());
    }
}
