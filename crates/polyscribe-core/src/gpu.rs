// SPDX-License-Identifier: MIT

//! GPU device detection through ggml's backend registry.
//!
//! ggml keeps a registry of the compute backends that are compiled into the
//! binary together with the devices those backends expose. whisper.cpp picks
//! a GPU out of that same registry, so asking ggml is the authoritative way
//! to find out whether a usable device exists. Guessing from environment
//! variables, device names or log output would be less reliable: the
//! registry reflects what the runtime actually found.
//!
//! The registry is filled in when it is first used, so the first call here
//! also triggers the backend initialization (and its diagnostics).

/// A GPU compute backend that PolyScribe can request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA.
    Cuda,
    /// AMD ROCm/HIP.
    Hip,
}

impl GpuBackend {
    /// The name ggml uses for this backend in its registry.
    ///
    /// The HIP build of ggml reuses the CUDA backend sources, so the
    /// registry reports "ROCm" for AMD devices and "CUDA" for NVIDIA ones.
    // Only the probe below uses this, and the probe only exists in a
    // GPU-enabled build.
    #[cfg_attr(not(any(feature = "gpu-cuda", feature = "gpu-hip")), allow(dead_code))]
    fn registry_name(self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Hip => "ROCm",
        }
    }
}

/// A GPU device that ggml reports for a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// ggml's device name, for example `ROCm0`.
    pub name: String,
    /// The device description, for example the marketing name of the card.
    pub description: String,
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    // ggml's public backend registry API, see ggml/include/ggml-backend.h.
    // The definitions live in the ggml build that whisper-rs-sys links, so
    // they resolve when this crate is linked against it.
    unsafe extern "C" {
        pub fn ggml_backend_dev_count() -> usize;
        pub fn ggml_backend_dev_get(index: usize) -> *mut c_void;
        pub fn ggml_backend_dev_type(dev: *mut c_void) -> c_int;
        pub fn ggml_backend_dev_name(dev: *mut c_void) -> *const c_char;
        pub fn ggml_backend_dev_description(dev: *mut c_void) -> *const c_char;
        pub fn ggml_backend_dev_backend_reg(dev: *mut c_void) -> *mut c_void;
        pub fn ggml_backend_reg_name(reg: *mut c_void) -> *const c_char;
    }

    /// `GGML_BACKEND_DEVICE_TYPE_GPU` and `GGML_BACKEND_DEVICE_TYPE_IGPU`.
    ///
    /// whisper.cpp accepts both when it looks for a device, so the probe
    /// accepts them as well.
    pub const DEVICE_TYPE_GPU: c_int = 1;
    pub const DEVICE_TYPE_IGPU: c_int = 3;
}

// Only the probe below uses this, and the probe only exists in a
// GPU-enabled build.
#[cfg_attr(not(any(feature = "gpu-cuda", feature = "gpu-hip")), allow(dead_code))]
fn c_str_to_string(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// The GPU devices that ggml reports for the given backend.
///
/// The list is empty when the backend is not compiled into this binary, when
/// the runtime is not usable, or when no matching device is present.
pub fn gpu_devices(backend: GpuBackend) -> Vec<GpuDevice> {
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-hip")))]
    {
        let _ = backend;
        Vec::new()
    }

    #[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
    {
        let mut found = Vec::new();
        let count = unsafe { ffi::ggml_backend_dev_count() };
        for index in 0..count {
            let dev = unsafe { ffi::ggml_backend_dev_get(index) };
            if dev.is_null() {
                continue;
            }
            let dev_type = unsafe { ffi::ggml_backend_dev_type(dev) };
            if dev_type != ffi::DEVICE_TYPE_GPU && dev_type != ffi::DEVICE_TYPE_IGPU {
                continue;
            }
            let reg = unsafe { ffi::ggml_backend_dev_backend_reg(dev) };
            if reg.is_null() {
                continue;
            }
            let reg_name = c_str_to_string(unsafe { ffi::ggml_backend_reg_name(reg) });
            if reg_name != backend.registry_name() {
                continue;
            }
            found.push(GpuDevice {
                name: c_str_to_string(unsafe { ffi::ggml_backend_dev_name(dev) }),
                description: c_str_to_string(unsafe { ffi::ggml_backend_dev_description(dev) }),
            });
        }
        found
    }
}

/// Whether at least one device for the given backend is present.
pub fn gpu_device_available(backend: GpuBackend) -> bool {
    !gpu_devices(backend).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_names_match_ggml() {
        // ggml reports the ROCm backend under this name, which is what the
        // probe matches against.
        assert_eq!(GpuBackend::Hip.registry_name(), "ROCm");
        assert_eq!(GpuBackend::Cuda.registry_name(), "CUDA");
    }

    #[test]
    fn test_probe_reports_no_devices_without_a_gpu_feature() {
        // A CPU-only build has no GPU backend compiled in, so the probe must
        // not report any device.
        #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-hip")))]
        {
            assert!(gpu_devices(GpuBackend::Cuda).is_empty());
            assert!(gpu_devices(GpuBackend::Hip).is_empty());
            assert!(!gpu_device_available(GpuBackend::Cuda));
            assert!(!gpu_device_available(GpuBackend::Hip));
        }
    }
}
