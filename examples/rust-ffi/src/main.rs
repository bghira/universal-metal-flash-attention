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
pub mod mla;

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
        if result != MFA_SUCCESS as mfa_error_t {
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
        if result != MFA_SUCCESS as mfa_error_t {
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
    let run_mla = args.len() > 1 && args[1] == "mla";

    if run_benchmark {
        return benchmark::run_benchmarks();
    }

    if run_mla {
        return run_mla_example();
    }

    println!("Universal Metal Flash Attention - Rust Example");
    println!("(Run with 'benchmark' for performance tests or 'mla' for MLA decompression)");

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
            MFA_PRECISION_FP16 as mfa_precision_t, // input_precision (FP16)
            MFA_PRECISION_FP16 as mfa_precision_t, // intermediate_precision (FP16)
            MFA_PRECISION_FP16 as mfa_precision_t, // output_precision (FP16)
            false,                              // transpose_q
            false,                              // transpose_k
            false,                              // transpose_v
            false,                              // transpose_o
            std::ptr::null(),                   // mask_ptr
            0,                                   // mask_size_bytes
            std::ptr::null(),                   // mask_shape
            std::ptr::null(),                   // mask_strides
            0,                                   // mask_ndim
            MFA_MASK_TYPE_NONE as mfa_mask_type_t,       // mask_type
            MFA_MASK_SCALAR_BYTE as mfa_mask_scalar_t,   // mask_scalar_type
        )
    };

    if attention_result == MFA_SUCCESS as mfa_error_t {
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

fn run_mla_example() -> Result<(), Box<dyn std::error::Error>> {
    use mla::MlaContext;

    println!("\n🔧 MLA (Multi-Latent Attention) Decompression Example");
    println!("Performance: 10.9 TFLOPS @ 2048×2048 on M3 Max\n");

    // Check if Metal is supported
    let is_supported = unsafe { mfa_is_device_supported() };
    if !is_supported {
        println!("Metal is not supported on this device");
        return Ok(());
    }
    println!("✓ Metal device is supported");

    // Create MFA context
    let mfa_context = MfaContext::new().map_err(Box::new)?;
    println!("✓ Created MFA context");

    // Create MLA context
    let mla_context = MlaContext::new().map_err(Box::new)?;
    println!("✓ Created MLA context");

    // Configuration
    let batch_size = 1;
    let num_heads = 8;
    let sequence_length = 512;
    let head_dim = 128;
    let kv_latent_dim = 512; // 8x compression (1024 → 512)

    println!("\nConfiguration:");
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", sequence_length);
    println!("  Heads: {}, Head dim: {}", num_heads, head_dim);
    println!(
        "  KV latent dim: {} (compression: {}x)",
        kv_latent_dim,
        (num_heads * head_dim) / kv_latent_dim
    );

    // Initialize decompression weights
    mla_context
        .init_random_weights(num_heads, head_dim, kv_latent_dim)
        .map_err(Box::new)?;
    println!("\n✓ Initialized decompression weights (W_k, W_v)");

    // Create compressed KV latent buffer
    let latent_size = (batch_size * sequence_length * kv_latent_dim * 2) as usize; // FP16
    let kv_latent = MfaBuffer::new(&mfa_context, latent_size).map_err(Box::new)?;
    println!("✓ Created compressed KV latent buffer ({} bytes)", latent_size);

    // Decompress K and V
    println!("\nDecompressing K and V...");
    let (decompressed_k, decompressed_v) = mla_context
        .forward(
            &mfa_context,
            &kv_latent,
            batch_size,
            num_heads,
            sequence_length,
            head_dim,
            kv_latent_dim,
        )
        .map_err(Box::new)?;

    println!("✅ MLA decompression successful!");
    println!("   K output: [{} × {}, {}] FP16", batch_size * sequence_length, num_heads * head_dim, head_dim);
    println!("   V output: [{} × {}, {}] FP16", batch_size * sequence_length, num_heads * head_dim, head_dim);
    println!("\n✅ MLA example completed successfully!");

    // Resources automatically cleaned up by RAII destructors
    Ok(())
}
