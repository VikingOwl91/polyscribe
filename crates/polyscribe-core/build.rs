// SPDX-License-Identifier: MIT
// Move original build.rs behavior into core crate
fn main() {
    // Only run special build steps when gpu-vulkan feature is enabled.
    let vulkan_enabled = std::env::var("CARGO_FEATURE_GPU_VULKAN").is_ok();
    if !vulkan_enabled {
        return;
    }
    println!("cargo:rerun-if-changed=extern/whisper.cpp");
    println!(
        "cargo:warning=Building with gpu-vulkan: ensure Vulkan SDK/loader are installed. Future versions will compile whisper.cpp via CMake."
    );
}
