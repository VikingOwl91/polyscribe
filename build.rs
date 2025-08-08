// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

fn main() {
    // Only run special build steps when gpu-vulkan feature is enabled.
    let vulkan_enabled = std::env::var("CARGO_FEATURE_GPU_VULKAN").is_ok();
    if !vulkan_enabled {
        return;
    }
    // Placeholder: In a full implementation, we would invoke CMake for whisper.cpp with GGML_VULKAN=1.
    // For now, emit a helpful note. Build will proceed; runtime Vulkan backend returns an explanatory error.
    println!("cargo:rerun-if-changed=extern/whisper.cpp");
    println!(
        "cargo:warning=Building with gpu-vulkan: ensure Vulkan SDK/loader are installed. Future versions will compile whisper.cpp via CMake."
    );
}
