// Safe Rust wrappers for MLA (Multi-Latent Attention) decompression
//
// Provides RAII wrappers and safe abstractions over the C FFI

use std::ffi::c_void;
use std::ptr;

use crate::bindings::*;
use crate::{mfa_error_from_code, MfaBuffer, MfaContext, MfaError};

/// RAII wrapper for MLA decompression context
///
/// MLA (Multi-Latent Attention) compresses KV cache from [batch, seq, num_heads × head_dim]
/// to [batch, seq, kv_latent_dim]. This context manages decompression weights and GEMM kernels.
///
/// Performance: 10.9 TFLOPS @ 2048×2048 on M3 Max
pub struct MlaContext(*mut c_void);

impl MlaContext {
    /// Create a new MLA decompression context
    pub fn new() -> Result<Self, MfaError> {
        let mut context: *mut c_void = ptr::null_mut();
        let result = unsafe { mfa_mla_create_context(&mut context) };
        if result != MFA_SUCCESS as mfa_error_t {
            Err(mfa_error_from_code(result))
        } else {
            Ok(MlaContext(context))
        }
    }

    /// Initialize random decompression weights for testing
    ///
    /// For production use, load pre-trained weights with `load_weights` instead.
    pub fn init_random_weights(
        &self,
        num_heads: u32,
        head_dim: u32,
        kv_latent_dim: u32,
    ) -> Result<(), MfaError> {
        let result = unsafe {
            mfa_mla_init_weights(self.as_ptr(), num_heads, head_dim, kv_latent_dim)
        };
        if result != MFA_SUCCESS as mfa_error_t {
            Err(mfa_error_from_code(result))
        } else {
            Ok(())
        }
    }

    /// Load pre-trained decompression weights
    ///
    /// # Arguments
    /// * `wk` - Weight matrix for K decompression [kv_latent_dim, num_heads × head_dim]
    /// * `wv` - Weight matrix for V decompression [kv_latent_dim, num_heads × head_dim]
    pub fn load_weights(&self, wk: &MfaBuffer, wv: &MfaBuffer) -> Result<(), MfaError> {
        let result =
            unsafe { mfa_mla_load_weights(self.as_ptr(), wk.as_ptr(), wv.as_ptr()) };
        if result != MFA_SUCCESS as mfa_error_t {
            Err(mfa_error_from_code(result))
        } else {
            Ok(())
        }
    }

    /// Decompress KV latent representations into full K and V matrices
    ///
    /// Performs optimized GEMM operations:
    /// - K = KV_latent @ W_k
    /// - V = KV_latent @ W_v
    ///
    /// # Arguments
    /// * `mfa_context` - MFA context (for command queue)
    /// * `kv_latent` - Input compressed KV buffer [batch × seq, kv_latent_dim]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `sequence_length` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `kv_latent_dim` - Compressed latent dimension
    ///
    /// # Returns
    /// Tuple of (decompressed_k, decompressed_v) buffers
    pub fn forward(
        &self,
        mfa_context: &MfaContext,
        kv_latent: &MfaBuffer,
        batch_size: u32,
        num_heads: u32,
        sequence_length: u32,
        head_dim: u32,
        kv_latent_dim: u32,
    ) -> Result<(MfaBuffer, MfaBuffer), MfaError> {
        let mut decompressed_k: *mut c_void = ptr::null_mut();
        let mut decompressed_v: *mut c_void = ptr::null_mut();

        let result = unsafe {
            mfa_mla_forward(
                self.as_ptr(),
                mfa_context.as_ptr(),
                kv_latent.as_ptr(),
                &mut decompressed_k,
                &mut decompressed_v,
                batch_size,
                num_heads,
                sequence_length,
                head_dim,
                kv_latent_dim,
            )
        };

        if result != MFA_SUCCESS as mfa_error_t {
            Err(mfa_error_from_code(result))
        } else {
            // Wrap the output buffers
            Ok((MfaBuffer(decompressed_k), MfaBuffer(decompressed_v)))
        }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for MlaContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mfa_mla_destroy_context(self.0) };
        }
    }
}

// Ensure MlaContext is Send + Sync (safe because Metal handles thread-safety)
unsafe impl Send for MlaContext {}
unsafe impl Sync for MlaContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_context_creation() {
        let result = MlaContext::new();
        assert!(result.is_ok(), "Should create MLA context successfully");
    }

    #[test]
    fn test_mla_init_weights() {
        let mla_ctx = MlaContext::new().expect("Failed to create MLA context");
        let result = mla_ctx.init_random_weights(8, 128, 512);
        assert!(
            result.is_ok(),
            "Should initialize random weights successfully"
        );
    }

    #[test]
    fn test_mla_decompression() -> Result<(), Box<dyn std::error::Error>> {
        // Create contexts
        let mfa_ctx = MfaContext::new()?;
        let mla_ctx = MlaContext::new()?;

        // Configuration
        let batch_size = 1;
        let num_heads = 8;
        let sequence_length = 16;
        let head_dim = 128;
        let kv_latent_dim = 512;

        // Initialize weights
        mla_ctx.init_random_weights(num_heads, head_dim, kv_latent_dim)?;

        // Create compressed KV latent buffer
        let latent_size = (batch_size * sequence_length * kv_latent_dim * 2) as usize; // FP16
        let kv_latent = MfaBuffer::new(&mfa_ctx, latent_size)?;

        // Decompress
        let (decompressed_k, decompressed_v) = mla_ctx.forward(
            &mfa_ctx,
            &kv_latent,
            batch_size,
            num_heads,
            sequence_length,
            head_dim,
            kv_latent_dim,
        )?;

        // Verify output buffers are valid
        assert!(!decompressed_k.as_ptr().is_null());
        assert!(!decompressed_v.as_ptr().is_null());

        println!("✅ MLA decompression test passed!");
        Ok(())
    }
}
