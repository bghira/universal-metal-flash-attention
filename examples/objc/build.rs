use std::process::Command;

fn main() {
    // Build Universal Metal Flash Attention first
    let swift_output = Command::new("swift")
        .args(&[
            "build",
            "--package-path", "../..",
            "--configuration", "debug"  // Use debug build for consistency
        ])
        .output()
        .expect("Failed to build Universal Metal Flash Attention");

    if !swift_output.status.success() {
        println!("Universal MFA build stderr: {}", String::from_utf8_lossy(&swift_output.stderr));
        panic!("Failed to build Universal Metal Flash Attention");
    }

    // Link against the Universal Metal Flash Attention libraries
    println!("cargo:rustc-link-search=native=../../.build/debug");
    println!("cargo:rustc-link-lib=dylib=MFAFFI");  // C FFI library contains everything
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");

    // Compile our simplified Objective-C bridge (using C FFI like Rust)
    cc::Build::new()
        .file("simple_bridge.m")
        .include(".")
        .include("../../include")  // Include C FFI headers
        .flag("-fobjc-arc") // Enable ARC
        .flag("-O3")
        .compile("simple_objc_bridge");

    // Tell cargo to recompile if files change
    println!("cargo:rerun-if-changed=simple_bridge.m");
    println!("cargo:rerun-if-changed=simple_bridge.h");
}
