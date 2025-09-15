use objc::runtime::{Class, Object, BOOL, YES, NO};
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::{c_void, CStr};

// Objective-C types
type id = *mut Object;
type MTLDevice = id;
type MTLBuffer = id;

// Simplified Rust wrapper using SimpleBridge that forwards to MFABridge.swift
struct SimpleBridge {
    objc_instance: id,
}

impl SimpleBridge {
    fn new(device: MTLDevice) -> Result<Self, String> {
        unsafe {
            // Use SimpleBridge which forwards to MFABridge.swift (same approach as Rust->FFI->Swift)
            let cls = class!(SimpleBridge);
            let instance: id = msg_send![cls, alloc];
            let instance: id = msg_send![instance, initWithDevice: device];

            if instance.is_null() {
                return Err("Failed to create SimpleBridge instance".to_string());
            }

            Ok(SimpleBridge {
                objc_instance: instance,
            })
        }
    }

    fn create_buffer(&self, size: usize) -> MTLBuffer {
        unsafe {
            msg_send![self.objc_instance, createBufferWithSize: size]
        }
    }

    // Match the Rust->FFI->Swift approach: delegate to MFABridge.swift
    fn run_attention_forward(&self,
                           q_buffer: MTLBuffer,
                           k_buffer: MTLBuffer,
                           v_buffer: MTLBuffer,
                           o_buffer: MTLBuffer,
                           batch_size: u32,
                           seq_len_q: u32,
                           seq_len_kv: u32,
                           num_heads: u32,
                           head_dim: u16,
                           softmax_scale: f32,
                           causal: bool,
                           input_precision: i32,
                           intermediate_precision: i32,
                           output_precision: i32) -> f64 {
        unsafe {
            let causal_bool: BOOL = if causal { YES } else { NO };
            let transpose_q: BOOL = NO;
            let transpose_k: BOOL = NO;
            let transpose_v: BOOL = NO;
            let transpose_o: BOOL = NO;

            // Call the simplified bridge method
            msg_send![self.objc_instance,
                     attentionForwardWithQ: q_buffer
                     k: k_buffer
                     v: v_buffer
                     out: o_buffer
                     batchSize: batch_size
                     seqLenQ: seq_len_q
                     seqLenKV: seq_len_kv
                     numHeads: num_heads
                     headDim: head_dim
                     softmaxScale: softmax_scale
                     causal: causal_bool
                     inputPrecision: input_precision
                     intermediatePrecision: intermediate_precision
                     outputPrecision: output_precision
                     transposeQ: transpose_q
                     transposeK: transpose_k
                     transposeV: transpose_v
                     transposeO: transpose_o]
        }
    }

    fn get_gpu_latency(&self) -> f64 {
        unsafe {
            msg_send![self.objc_instance, getGpuLatency]
        }
    }

    fn get_version(&self) -> String {
        unsafe {
            let ns_string: id = msg_send![self.objc_instance, getVersion];
            let c_str: *const i8 = msg_send![ns_string, UTF8String];
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }
}

impl Drop for SimpleBridge {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.objc_instance, release];
        }
    }
}

// Benchmark result structure
struct BenchmarkResult {
    pub mean_time_ms: f64,
    pub ginstrs_per_second: f64,
}

impl BenchmarkResult {
    fn print(&self, config: &str) {
        println!("{:12} {:>10.0} GINSTRS/s", config, self.ginstrs_per_second);
    }
}

fn benchmark_attention(bridge: &SimpleBridge, seq_len: usize, head_dim: usize) -> Result<BenchmarkResult, String> {
    let element_size = 4; // FP32 = 4 bytes (to match Rust->FFI->Swift)
    let buffer_size = seq_len * head_dim * element_size;

    // Create buffers
    let q_buffer = bridge.create_buffer(buffer_size);
    let k_buffer = bridge.create_buffer(buffer_size);
    let v_buffer = bridge.create_buffer(buffer_size);
    let o_buffer = bridge.create_buffer(buffer_size);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Warmup (3 runs - match Rust approach)
    for _ in 0..3 {
        let result = bridge.run_attention_forward(
            q_buffer, k_buffer, v_buffer, o_buffer,
            1,                              // batch_size
            seq_len as u32,                 // seq_len_q
            seq_len as u32,                 // seq_len_kv
            1,                              // num_heads
            head_dim as u16,                // head_dim
            scale,                          // softmax_scale
            false,                          // causal
            1,                              // input_precision (FP32)
            1,                              // intermediate_precision (FP32)
            1,                              // output_precision (FP32)
        );
        if result < 0.0 {
            return Err("Warmup run failed".to_string());
        }
    }

    // Benchmark runs (5 runs - match Rust approach)
    let mut times = Vec::new();
    for _ in 0..5 {
        let result = bridge.run_attention_forward(
            q_buffer, k_buffer, v_buffer, o_buffer,
            1,                              // batch_size
            seq_len as u32,                 // seq_len_q
            seq_len as u32,                 // seq_len_kv
            1,                              // num_heads
            head_dim as u16,                // head_dim
            scale,                          // softmax_scale
            false,                          // causal
            1,                              // input_precision (FP32)
            1,                              // intermediate_precision (FP32)
            1,                              // output_precision (FP32)
        );

        if result < 0.0 {
            return Err("Benchmark run failed".to_string());
        }

        // Get pure GPU timing (zero-overhead measurement like Rust->FFI->Swift)
        let gpu_latency_seconds = bridge.get_gpu_latency();
        times.push(gpu_latency_seconds * 1000.0); // Convert to ms
    }

    let mean_time_ms = times.iter().sum::<f64>() / times.len() as f64;

    // Calculate GINSTRS/s using EXACT Swift formula: (2*D + 5) * N²
    // MFABridge dispatches 5x per call like native Swift, so multiply operations by 5
    let operations = (2 * head_dim + 5) * seq_len * seq_len * 5;
    let ginstrs_per_second = (operations as f64 / 1e9) / (mean_time_ms / 1000.0);

    Ok(BenchmarkResult {
        mean_time_ms,
        ginstrs_per_second,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Simplified Metal Flash Attention - Objective-C→Swift Bridge");
    println!("================================================================");
    println!("(Matching Rust→FFI→Swift approach exactly)");

    // Create Metal device
    let mtl_device = unsafe {
        let mtl_device_class = class!(MTLCreateSystemDefaultDevice);
        let device: id = msg_send![mtl_device_class, call];
        if device.is_null() {
            return Err("Failed to create Metal device".into());
        }
        device
    };

    // Create SimpleBridge (forwards to MFABridge.swift same as Rust->FFI->Swift)
    let bridge = SimpleBridge::new(mtl_device)?;
    println!("✅ MFA version: {}", bridge.get_version());
    println!("✅ Metal device is supported");

    // EXACT Rust benchmark configurations: N=1024, D=[16,64,256]
    let configs = vec![
        (1024, 16),
        (1024, 64),
        (1024, 256),
    ];

    println!("\n📊 Apples-to-Apples Performance Comparison");
    println!("--------------------------------------------------");
    println!("Config         FWD (GINSTRS/s)");
    println!("--------------------------------------------------");

    let mut results = Vec::new();

    for (seq_len, head_dim) in configs {
        match benchmark_attention(&bridge, seq_len, head_dim) {
            Ok(result) => {
                println!("{}x{:<8} {:>8.0}", seq_len, head_dim, result.ginstrs_per_second);
                results.push(result);
            }
            Err(e) => {
                println!("{}x{:<8} ERROR: {}", seq_len, head_dim, e);
            }
        }
    }

    println!("--------------------------------------------------");

    // Performance analysis
    if !results.is_empty() {
        let best_result = results.iter()
            .max_by(|a, b| a.ginstrs_per_second.partial_cmp(&b.ginstrs_per_second).unwrap())
            .unwrap();

        println!("\n📈 Performance Analysis:");
        println!("   • Peak Performance: {:.1} GINSTRS/s", best_result.ginstrs_per_second);

        // Compare to target
        if best_result.ginstrs_per_second < 4400.0 {
            let gap = 4400.0 / best_result.ginstrs_per_second;
            println!("   • Gap to 4400 GINSTRS/s target: {:.1}x", gap);
        } else {
            println!("   ✅ Target 4400 GINSTRS/s achieved!");
        }

        println!("   • Direct Objective-C → MFABridge.swift calls");
        println!("   • Uses same caching & dispatch patterns as Rust→FFI→Swift");
        println!("   • Zero-copy buffer management");
        println!("   • GPU timing eliminates CPU overhead");
    }

    Ok(())
}
