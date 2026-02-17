use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
pub fn slugify(text: &str) -> String {
    text.to_lowercase().replace(' ', "-").replace(|c: char| !c.is_alphanumeric() && c != '-', "")
}

#[allow(dead_code)]
fn run_command(cmd: &str) -> Option<String> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

#[allow(dead_code)]
pub fn get_git_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    if let Some(rev) = run_command("git rev-parse HEAD") {
        info.insert("revision".into(), rev);
    }
    info
}

#[allow(dead_code)]
pub fn get_system_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("os".into(), std::env::consts::OS.into());
    info.insert("arch".into(), std::env::consts::ARCH.into());
    info
}

#[allow(dead_code)]
pub fn generate_header() -> String {
    "NANOCHAT REPORT".into()
}

#[allow(dead_code)]
pub const EXPECTED_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "tokenizer.json",
];

pub struct Report {
    #[allow(dead_code)]
    report_dir: PathBuf,
}

impl Report {
    pub fn new(report_dir: impl AsRef<Path>) -> Self {
        Self { report_dir: report_dir.as_ref().to_path_buf() }
    }

    pub fn log(&self, _section: &str, _data: &[ReportItem]) {
        // Mock implementation
    }

    #[allow(dead_code)]
    pub fn generate(&self) -> PathBuf {
        PathBuf::from("report.json")
    }

    #[allow(dead_code)]
    pub fn reset(&self) {
        // Mock reset
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum ReportItem {
    Text { key: String, value: String },
    Value { key: String, value: f64 },
}

#[allow(dead_code)]
pub struct DummyReport;

impl DummyReport {
    #[allow(dead_code)]
    pub fn log(&self, _section: &str, _data: &[ReportItem]) {}
    #[allow(dead_code)]
    pub fn reset(&self) {}
}

pub fn get_report() -> Report {
    Report::new("reports")
}
