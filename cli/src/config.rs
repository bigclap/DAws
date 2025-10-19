use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use trainer::OfflineTrainerConfig;

/// Settings driving the `train` command.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TrainSettings {
    /// Optional dataset of decoder samples consumed during validation.
    #[serde(default)]
    pub dataset: Option<PathBuf>,
    /// Location where the training checkpoint summary is written.
    #[serde(default = "TrainSettings::default_checkpoint")]
    pub checkpoint: PathBuf,
    /// Optional flamegraph destination collected while training.
    #[serde(default = "TrainSettings::default_profile_output")]
    pub profile_output: Option<PathBuf>,
    /// Offline trainer hyper-parameters.
    #[serde(default)]
    pub trainer: OfflineTrainerConfig,
}

impl TrainSettings {
    fn default_checkpoint() -> PathBuf {
        PathBuf::from("checkpoints/offline.ckpt")
    }

    fn default_profile_output() -> Option<PathBuf> {
        Some(PathBuf::from("profiles/train.flamegraph.svg"))
    }
}

impl Default for TrainSettings {
    fn default() -> Self {
        Self {
            dataset: None,
            checkpoint: Self::default_checkpoint(),
            profile_output: Self::default_profile_output(),
            trainer: OfflineTrainerConfig::default(),
        }
    }
}

/// Settings powering the `eval` command.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EvalSettings {
    /// Optional dataset used for validation.
    #[serde(default)]
    pub dataset: Option<PathBuf>,
    /// Checkpoint consumed by evaluation (purely informational in the demo).
    #[serde(default = "EvalSettings::default_checkpoint")]
    pub checkpoint: PathBuf,
    /// Path where the evaluation report is written.
    #[serde(default = "EvalSettings::default_report")]
    pub report: PathBuf,
    /// Optional flamegraph destination for evaluation runs.
    #[serde(default = "EvalSettings::default_profile_output")]
    pub profile_output: Option<PathBuf>,
    /// Offline trainer configuration reused when producing validation metrics.
    #[serde(default)]
    pub trainer: OfflineTrainerConfig,
}

impl EvalSettings {
    fn default_checkpoint() -> PathBuf {
        PathBuf::from("checkpoints/offline.ckpt")
    }

    fn default_report() -> PathBuf {
        PathBuf::from("reports/eval_summary.txt")
    }

    fn default_profile_output() -> Option<PathBuf> {
        Some(PathBuf::from("profiles/eval.flamegraph.svg"))
    }
}

impl Default for EvalSettings {
    fn default() -> Self {
        Self {
            dataset: None,
            checkpoint: Self::default_checkpoint(),
            report: Self::default_report(),
            profile_output: Self::default_profile_output(),
            trainer: OfflineTrainerConfig::default(),
        }
    }
}

/// Settings consumed by the `infer` command.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InferSettings {
    /// XOR inputs evaluated during inference.
    #[serde(default = "InferSettings::default_inputs")]
    pub inputs: Vec<String>,
    /// Optional checkpoint hint surfaced to the user.
    #[serde(default)]
    pub checkpoint: Option<PathBuf>,
    /// Optional profiling output.
    #[serde(default = "InferSettings::default_profile_output")]
    pub profile_output: Option<PathBuf>,
}

impl InferSettings {
    fn default_inputs() -> Vec<String> {
        vec![
            "0 0".to_string(),
            "0 1".to_string(),
            "1 0".to_string(),
            "1 1".to_string(),
        ]
    }

    fn default_profile_output() -> Option<PathBuf> {
        Some(PathBuf::from("profiles/infer.flamegraph.svg"))
    }
}

impl Default for InferSettings {
    fn default() -> Self {
        Self {
            inputs: Self::default_inputs(),
            checkpoint: None,
            profile_output: Self::default_profile_output(),
        }
    }
}

/// Settings for the dedicated `profile` command.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ProfileSettings {
    /// XOR inputs executed while collecting the CPU profile.
    #[serde(default = "ProfileSettings::default_inputs")]
    pub inputs: Vec<String>,
    /// Target location for the generated flamegraph.
    #[serde(default = "ProfileSettings::default_output")]
    pub profile_output: PathBuf,
}

impl ProfileSettings {
    fn default_inputs() -> Vec<String> {
        InferSettings::default_inputs()
    }

    fn default_output() -> PathBuf {
        PathBuf::from("profiles/runtime.flamegraph.svg")
    }
}

impl Default for ProfileSettings {
    fn default() -> Self {
        Self {
            inputs: Self::default_inputs(),
            profile_output: Self::default_output(),
        }
    }
}

/// Loads TOML settings for the requested command, falling back to defaults when missing.
pub fn load_settings<T>(command: &str, explicit: Option<PathBuf>) -> Result<T>
where
    T: DeserializeOwned + Default,
{
    let (candidate, explicit_provided) = match explicit {
        Some(path) => (path, true),
        None => (PathBuf::from(format!("{command}.toml")), false),
    };

    if candidate.exists() {
        let raw = std::fs::read_to_string(&candidate).with_context(|| {
            format!(
                "failed to read configuration for `{command}` from {}",
                candidate.display()
            )
        })?;
        let parsed = toml::from_str(&raw).with_context(|| {
            format!(
                "failed to parse TOML configuration for `{command}` at {}",
                candidate.display()
            )
        })?;
        Ok(parsed)
    } else if explicit_provided {
        bail!(
            "configuration file for `{command}` not found at {}",
            candidate.display()
        );
    } else {
        Ok(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn load_settings_returns_default_when_missing() {
        let settings: TrainSettings = load_settings("train", None).expect("default settings");
        assert_eq!(settings, TrainSettings::default());
    }

    #[test]
    fn load_settings_errors_for_missing_explicit_path() {
        let result: Result<TrainSettings> =
            load_settings("train", Some(PathBuf::from("definitely_missing.toml")));
        assert!(result.is_err());
    }

    #[test]
    fn load_settings_parses_toml_payload() {
        let file = NamedTempFile::new().expect("temp file");
        let path = file.into_temp_path();
        std::fs::write(
            &path,
            r#"inputs = ["1 0"]
checkpoint = "ckpts/demo.ckpt"
profile_output = "profiles/demo.svg"
"#,
        )
        .expect("write config");

        let settings: InferSettings =
            load_settings("infer", Some(path.to_path_buf())).expect("parsed settings");
        assert_eq!(settings.inputs, vec!["1 0".to_string()]);
        assert_eq!(settings.checkpoint, Some(PathBuf::from("ckpts/demo.ckpt")));
        assert_eq!(
            settings.profile_output,
            Some(PathBuf::from("profiles/demo.svg"))
        );
    }
}
