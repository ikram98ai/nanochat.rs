//! Sandboxed execution utilities for running Python code that comes out of an LLM
//! (used by HumanEval).
//!
//! The code runs in a fresh Python subprocess. What is covered:
//! - Each execution runs in its own process (killed hard by the parent on timeout)
//! - A fresh interpreter: no access to the parent process memory, and a scrubbed environment
//! - Memory limits are enforced via rlimits (256MB by default)
//! - stdout and stderr are captured, stdin is disabled
//! - Code runs in a temporary directory that is deleted afterwards
//! - Destructive functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)
//!
//! What is not covered:
//! - Not a true security sandbox
//! - Network access is not blocked (e.g. sockets could be opened)
//! - Python's dynamic features (e.g. ctypes) could bypass restrictions
//! - No kernel-level isolation (no seccomp, no containers, no virtualization)
//!
//! Overall this sandbox is good for evaluation of generated code and protects against
//! accidental destructive behavior, but it is not safe against malicious adversarial code.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Result of executing Python code in a sandbox.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub timeout: bool,
    pub memory_exceeded: bool,
}

/// 256MB default memory limit
pub const DEFAULT_MAXIMUM_MEMORY_BYTES: u64 = 256 * 1024 * 1024;

// The guard runs in the subprocess before the untrusted code. It applies the
// resource limits and disables destructive functions to protect against
// accidents (a fork bomb, deleting files, killing other processes, ...).
// It is trivially bypassable by adversarial code, see the module docs above.
const GUARD: &str = r#"
import faulthandler, builtins, os, shutil, subprocess, sys
maximum_memory_bytes = {maximum_memory_bytes}
if maximum_memory_bytes is not None and sys.platform != "darwin":
    # (the resource limit calls seem to fail on macOS, skip them there)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
faulthandler.disable()
builtins.exit = None
builtins.quit = None
builtins.help = None
os.environ["OMP_NUM_THREADS"] = "1"
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
             "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
             "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
             "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    setattr(os, name, None)
for name in ("rmtree", "move", "chown"):
    setattr(shutil, name, None)
subprocess.Popen = None
for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[name] = None
"#;

/// Render a string as an ASCII-only Python string literal (the equivalent of
/// embedding `{code!r}` in the reference).
fn python_str_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c if c.is_ascii() => out.push(c),
            c if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push('"');
    out
}

/// Locate the python3 interpreter on the parent's PATH (the child runs with a
/// scrubbed environment, so the lookup has to happen here).
fn find_python() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join("python3"))
        .find(|p| p.is_file())
}

/// A fresh private directory for one execution; removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> std::io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!(
            "nanochat_exec_{}_{}_{}",
            std::process::id(),
            nanos,
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path)?;
        Ok(Self(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn failure(error: String) -> ExecutionResult {
    ExecutionResult {
        success: false,
        stdout: String::new(),
        stderr: String::new(),
        error: Some(error),
        timeout: false,
        memory_exceeded: false,
    }
}

/// Execute Python code in a sandboxed environment with the default 256MB memory
/// limit. Success means the program ran to completion without raising.
pub fn execute_code(code: &str, timeout_secs: f64) -> ExecutionResult {
    execute_code_with_limits(code, timeout_secs, Some(DEFAULT_MAXIMUM_MEMORY_BYTES))
}

/// Execute Python code in a sandboxed environment.
///
/// - `timeout_secs`: maximum execution time in seconds
/// - `maximum_memory_bytes`: memory limit in bytes (None to disable)
pub fn execute_code_with_limits(
    code: &str,
    timeout_secs: f64,
    maximum_memory_bytes: Option<u64>,
) -> ExecutionResult {
    // the guard runs first, then the untrusted code (with fresh globals, as a string literal)
    let limit = maximum_memory_bytes.map_or("None".to_string(), |b| b.to_string());
    let guard = GUARD.replace("{maximum_memory_bytes}", &limit);
    let program = format!(
        "{guard}\nexec(compile({}, '<llm>', 'exec'), {{'__name__': '__main__'}})\n",
        python_str_literal(code)
    );

    let Some(python) = find_python() else {
        return failure("failed to launch python3: not found on PATH".to_string());
    };
    let tmpdir = match TempDir::new() {
        Ok(tmpdir) => tmpdir,
        Err(e) => return failure(format!("failed to create a temporary directory: {e}")),
    };

    let child = Command::new(python)
        .arg("-c")
        .arg(&program)
        .current_dir(&tmpdir.0) // writes land in the tempdir, deleted afterwards
        .env_clear() // scrub the environment
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(e) => return failure(format!("failed to launch python3: {e}")),
    };

    // Drain the pipes on separate threads so a chatty program cannot block on a full pipe
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut buf);
            }
            String::from_utf8_lossy(&buf).into_owned()
        })
    };
    let stdout_reader = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let stderr_reader = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );

    let deadline = Instant::now() + Duration::from_secs_f64(timeout_secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    // kill the child process on timeout
                    let _ = child.kill();
                    let _ = child.wait();
                    return ExecutionResult {
                        success: false,
                        stdout: String::new(),
                        stderr: String::new(),
                        error: Some("Execution timed out (process killed)".to_string()),
                        timeout: true,
                        memory_exceeded: false,
                    };
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                let _ = child.kill();
                return failure(e.to_string());
            }
        }
    };
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();

    let success = status.success();
    // the last line of the traceback identifies the exception, e.g. "TypeError: ..."
    let error = if success {
        None
    } else {
        Some(
            stderr
                .trim()
                .lines()
                .last()
                .unwrap_or("Execution failed")
                .to_string(),
        )
    };
    let memory_exceeded = stderr.contains("MemoryError");
    ExecutionResult {
        success,
        stdout,
        stderr,
        error,
        timeout: false,
        memory_exceeded,
    }
}
