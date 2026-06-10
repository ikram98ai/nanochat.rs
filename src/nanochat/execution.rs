//! Sandboxed execution of model-generated Python code (used by HumanEval).
//! The reference runs the code in a forked process with a time limit and
//! captured output; here the code runs in a python3 subprocess with the same
//! timeout semantics.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub timeout: bool,
}

/// Execute Python code with a time limit. Success means the program ran to
/// completion without raising (exit status 0).
pub fn execute_code(code: &str, timeout_secs: f64) -> ExecutionResult {
    let child = Command::new("python3")
        .arg("-c")
        .arg(code)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(e) => {
            return ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!("failed to launch python3: {e}"),
                timeout: false,
            }
        }
    };

    let deadline = Instant::now() + Duration::from_secs_f64(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut out) = child.stdout.take() {
                    let _ = out.read_to_string(&mut stdout);
                }
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut stderr);
                }
                return ExecutionResult { success: status.success(), stdout, stderr, timeout: false };
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return ExecutionResult {
                        success: false,
                        stdout: String::new(),
                        stderr: "Execution timed out".to_string(),
                        timeout: true,
                    };
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                return ExecutionResult {
                    success: false,
                    stdout: String::new(),
                    stderr: e.to_string(),
                    timeout: false,
                }
            }
        }
    }
}
