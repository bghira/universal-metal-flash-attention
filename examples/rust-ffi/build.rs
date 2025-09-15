use std::env;
use std::path::PathBuf;

fn main() {
    // The bindgen::Builder is the main entry point to bindgen, and lets you
    // build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("../../Sources/MFAFFI/include/mfa_ffi.h")
        // Tell cargo to invalidate the built crate whenever the wrapper changes
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Only generate bindings for our MFA functions, not system headers
        .allowlist_function("mfa_.*")
        .allowlist_type("mfa_.*")
        .allowlist_var("MFA_.*")
        // Improve type naming
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        // Don't generate layout tests
        .layout_tests(false)
        // Generate Debug and other useful traits
        .derive_debug(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .derive_partialeq(true)
        .derive_partialord(true)
        // Use core instead of std
        .use_core()
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Link against the MFA FFI library (dynamic)
    println!("cargo:rustc-link-search=native=../../.build/debug");
    println!("cargo:rustc-link-lib=dylib=MFAFFI");
}
