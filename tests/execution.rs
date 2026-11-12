//! Tests for the sandboxed Python execution used by HumanEval.
//! Each test adversarially checks one property of the sandbox
//! (mirrors tests/test_execution.py of the Python reference).

use std::time::Instant;

use nanochat::execution::execute_code;

#[test]
fn test_happy_path() {
    let result = execute_code("print('hello world')", 5.0);
    assert!(result.success, "{result:?}");
    assert_eq!(result.stdout, "hello world\n");
    assert!(result.error.is_none());
}

#[test]
fn test_exception_is_captured() {
    let result = execute_code("1/0", 5.0);
    assert!(!result.success);
    assert!(result.error.as_deref().unwrap().contains("ZeroDivisionError"), "{result:?}");
}

#[test]
fn test_timeout_kills_infinite_loop() {
    let start = Instant::now();
    let result = execute_code("while True: pass", 2.0);
    let elapsed = start.elapsed().as_secs_f64();
    assert!(result.timeout && !result.success);
    assert!(elapsed < 5.0, "process was not killed promptly ({elapsed:.1}s)");
}

#[test]
fn test_chatty_program_does_not_block() {
    // far more output than a pipe buffer holds: must finish, not time out
    let result = execute_code("print('x' * 2_000_000)", 5.0);
    assert!(result.success && !result.timeout, "{:?}", result.error);
    assert_eq!(result.stdout.len(), 2_000_001);
}

#[test]
#[cfg(not(target_os = "macos"))] // the resource limits are skipped on macOS
fn test_memory_limit() {
    // 1GB allocation against the 256MB default limit
    let result = execute_code("x = bytearray(1024 * 1024 * 1024)", 5.0);
    assert!(!result.success);
    assert!(result.memory_exceeded);
}

#[test]
fn test_destructive_functions_disabled() {
    for evil in [
        "import os; os.system('echo pwned')",
        "import shutil; shutil.rmtree('/tmp')",
        "import subprocess; subprocess.Popen(['ls'])",
        "import os; os.kill(1, 9)",
        "import os; os.fork()",
    ] {
        let result = execute_code(evil, 5.0);
        assert!(!result.success, "should have failed: {evil}");
    }
}

#[test]
fn test_stdin_disabled() {
    let result = execute_code("input()", 5.0);
    assert!(!result.success);
}

#[test]
fn test_writes_go_to_tempdir() {
    let result = execute_code(
        "open('landmine.txt', 'w').write('x'); print(open('landmine.txt').read())",
        5.0,
    );
    assert!(result.success && result.stdout == "x\n", "{result:?}");
    assert!(
        !std::path::Path::new("landmine.txt").exists(),
        "file leaked outside the sandbox tempdir"
    );
}

#[test]
fn test_environment_is_scrubbed() {
    std::env::set_var("FAKE_SECRET_KEY", "hunter2");
    let result = execute_code("import os; print('FAKE_SECRET_KEY' in os.environ)", 5.0);
    std::env::remove_var("FAKE_SECRET_KEY");
    assert!(result.success && result.stdout.trim() == "False", "{result:?}");
}

#[test]
fn test_tricky_string_content() {
    // quotes, backslashes, braces and non-ASCII must survive being embedded into the runner
    let tricky = r#"s = 'it\'s "quoted" \\ {braces} \n é 你好 🌍'; print(len(s))"#;
    let result = execute_code(tricky, 5.0);
    assert!(result.success, "{result:?}");
    assert_eq!(result.stdout.trim(), "33");
}

#[test]
fn test_fresh_globals() {
    // user code must not see the guard prelude's imports
    let result = execute_code("print('GUARD' in dir()); print(__name__)", 5.0);
    assert!(result.success, "{result:?}");
    assert_eq!(result.stdout, "False\n__main__\n");
}
