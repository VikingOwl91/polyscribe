// SPDX-License-Identifier: MIT

fn main() {
    let vulkan_enabled = std::env::var("CARGO_FEATURE_GPU_VULKAN").is_ok();
    println!("cargo:rerun-if-changed=extern/whisper.cpp");
    if !vulkan_enabled {
        println!(
            "cargo:warning=gpu-vulkan feature is disabled; skipping Vulkan-dependent build steps."
        );
        return;
    }
    println!(
        "cargo:warning=Building with gpu-vulkan: ensure Vulkan SDK/loader are installed. Future versions will compile whisper.cpp via CMake."
    );
}
