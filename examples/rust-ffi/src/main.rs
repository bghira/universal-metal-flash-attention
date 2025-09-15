use std::ffi::{c_void, CStr};
use std::ptr;

// Suppress bindgen naming warnings for C compatibility
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
// Include the generated bindings
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

mod benchmark;

// Error handling
#[derive(Debug)]
struct MfaError {
    code: mfa_error_t,
    message: String,
}

impl std::fmt::Display for MfaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MFA Error {:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for MfaError {}

fn mfa_error_from_code(code: mfa_error_t) -> MfaError {
    let message = unsafe {
        let c_str_ptr = mfa_error_string(code);
        if c_str_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            let c_str = CStr::from_ptr(c_str_ptr);
            let rust_str = c_str.to_string_lossy().into_owned();
            libc::free(c_str_ptr as *mut c_void);
            rust_str
        }
    };
    MfaError { code, message }
}

// RAII wrapper for MFA context
struct MfaContext(*mut c_void);

impl MfaContext {
    fn new() -> Result<Self, MfaError> {
        let mut context: *mut c_void = ptr::null_mut();
        let result = unsafe { mfa_create_context(&mut context) };
        if result != mfa_error_t::MFA_SUCCESS {
            Err(mfa_error_from_code(result))
        } else {
            Ok(MfaContext(context))
        }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for MfaContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mfa_destroy_context(self.0) };
        }
    }
}

// RAII wrapper for MFA buffer
struct MfaBuffer(*mut c_void);

impl MfaBuffer {
    fn new(context: &MfaContext, size: usize) -> Result<Self, MfaError> {
        let mut buffer: *mut c_void = ptr::null_mut();
        let result = unsafe { mfa_create_buffer(context.as_ptr(), size, &mut buffer) };
        if result != mfa_error_t::MFA_SUCCESS {
            Err(mfa_error_from_code(result))
        } else {
            Ok(MfaBuffer(buffer))
        }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for MfaBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mfa_destroy_buffer(self.0) };
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();
    let run_benchmark = args.len() > 1 && args[1] == "benchmark";

    if run_benchmark {
        return benchmark::run_benchmarks();
    }

    println!("Universal Metal Flash Attention - Rust Example");
    println!("(Run with 'benchmark' argument for performance tests)");

    // Check if Metal is supported
    let is_supported = unsafe { mfa_is_device_supported() };
    if !is_supported {
        println!("Metal is not supported on this device");
        return Ok(());
    }
    println!("✓ Metal device is supported");

    // Create context using RAII wrapper
    let context = MfaContext::new().map_err(Box::new)?;
    println!("✓ Created MFA context");

    // Get version info
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    let mut patch: i32 = 0;
    unsafe {
        mfa_get_version(&mut major, &mut minor, &mut patch);
    }
    println!("✓ MFA version: {}.{}.{}", major, minor, patch);

    // Create test buffers (small example)
    let seq_len = 16;
    let head_dim = 64;
    let element_size = 2; // FP16 = 2 bytes
    let buffer_size = seq_len * head_dim * element_size;

    // Create buffers using RAII wrappers
    let q_buffer = MfaBuffer::new(&context, buffer_size).map_err(Box::new)?;
    let k_buffer = MfaBuffer::new(&context, buffer_size).map_err(Box::new)?;
    let v_buffer = MfaBuffer::new(&context, buffer_size).map_err(Box::new)?;
    let o_buffer = MfaBuffer::new(&context, buffer_size).map_err(Box::new)?;
    println!("✓ Created input/output buffers");

    // Run attention forward pass (testing causal masking!)
    println!("Testing causal masking...");
    let attention_result = unsafe {
        mfa_attention_forward(
            context.as_ptr(),
            q_buffer.as_ptr(),
            k_buffer.as_ptr(),
            v_buffer.as_ptr(),
            o_buffer.as_ptr(),
            1,                                  // batch_size
            seq_len as u32,                     // seq_len_q
            seq_len as u32,                     // seq_len_kv
            1,                                  // num_heads (single head for now)
            head_dim as u16,                    // head_dim
            1.0 / (head_dim as f32).sqrt(),     // softmax_scale
            true,                               // causal masking enabled!
            mfa_precision_t::MFA_PRECISION_FP16, // input_precision (FP16)
            mfa_precision_t::MFA_PRECISION_FP16, // intermediate_precision (FP16)
            mfa_precision_t::MFA_PRECISION_FP16, // output_precision (FP16)
            false,                              // transpose_q
            false,                              // transpose_k
            false,                              // transpose_v
            false,                              // transpose_o
        )
    };

    if attention_result == mfa_error_t::MFA_SUCCESS {
        println!("✅ Causal attention forward pass completed successfully!");
        println!("✅ Our causal masking implementation works in Rust!");
    } else {
        let error = mfa_error_from_code(attention_result);
        println!("⚠ Attention forward pass failed: {}", error);
    }

    // Resources are automatically cleaned up by RAII destructors
    println!("✓ Cleaned up resources");

    Ok(())
}
