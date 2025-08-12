// Unix-only tests for with_suppressed_stderr restoring file descriptors
// Skip on Windows and non-Unix targets.

#![cfg(unix)]

use std::panic::{catch_unwind, AssertUnwindSafe};

fn stat_of_fd(fd: i32) -> (u64, u64) {
    unsafe {
        let mut st: libc::stat = std::mem::zeroed();
        let r = libc::fstat(fd, &mut st as *mut libc::stat);
        assert_eq!(r, 0, "fstat failed on fd {fd}");
        (st.st_dev as u64, st.st_ino as u64)
    }
}

fn stat_of_path(path: &str) -> (u64, u64) {
    use std::ffi::CString;
    unsafe {
        let c = CString::new(path).unwrap();
        let fd = libc::open(c.as_ptr(), libc::O_RDONLY);
        assert!(fd >= 0, "failed to open {path}");
        let s = stat_of_fd(fd);
        let _ = libc::close(fd);
        s
    }
}

#[test]
fn stderr_is_redirected_and_restored() {
    let before = stat_of_fd(2);
    let devnull = stat_of_path("/dev/null");

    // During the call, fd 2 should be /dev/null; after, restored to before
    polyscribe::with_suppressed_stderr(|| {
        let inside = stat_of_fd(2);
        assert_eq!(inside, devnull, "stderr should point to /dev/null during suppression");
        // This write should be suppressed
        eprintln!("this should be suppressed");
    });

    let after = stat_of_fd(2);
    assert_eq!(after, before, "stderr should be restored after suppression");
}

#[test]
fn stderr_is_restored_even_if_closure_panics() {
    let before = stat_of_fd(2);
    let res = catch_unwind(AssertUnwindSafe(|| {
        polyscribe::with_suppressed_stderr(|| {
            // Trigger a deliberate panic inside the closure
            panic!("boom inside with_suppressed_stderr");
        });
    }));
    assert!(res.is_err(), "expected panic to propagate");
    let after = stat_of_fd(2);
    assert_eq!(after, before, "stderr should be restored after panic");
}
