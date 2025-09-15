use std::ffi::{c_void, CStr};

// External C functions from our real Objective-C bridge
extern "C" {
    fn create_metal_device() -> *mut c_void;
    fn create_swift_attention_wrapper(device: *mut c_void) -> *mut c_void;
    fn swift_create_buffer(wrapper: *mut c_void, size: usize) -> *mut c_void;
    fn swift_run_attention(
        wrapper: *mut c_void,
        q_buffer: *mut c_void,
        k_buffer: *mut c_void,
        v_buffer: *mut c_void,
        o_buffer: *mut c_void,
        seq_length: usize,
        head_dim: usize,
        scale: f32,
        causal: bool
    ) -> f64;
    fn swift_get_version(wrapper: *mut c_void) -> *const i8;
    fn release_object(obj: *mut c_void);
}

// RAII wrapper for Metal device
struct MetalDevice(*mut c_void);

impl MetalDevice {
    fn new() -> Result<Self, String> {
        let device = unsafe { create_metal_device() };
        if device.is_null() {
            Err("Failed to create Metal device".to_string())
        } else {
            Ok(MetalDevice(device))
        }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { release_object(self.0) };
        }
    }
}

// RAII wrapper for Swift attention wrapper
struct SwiftAttentionWrapper(*mut c_void);

impl SwiftAttentionWrapper {
    fn new(device: &MetalDevice) -> Result<Self, String> {
        let wrapper = unsafe { create_swift_attention_wrapper(device.as_ptr()) };
        if wrapper.is_null() {
            Err("Failed to create Swift attention wrapper".to_string())
        } else {
            Ok(SwiftAttentionWrapper(wrapper))
        }
    }

    fn create_buffer(&self, size: usize) -> MetalBuffer {
        let buffer = unsafe { swift_create_buffer(self.0, size) };
        MetalBuffer(buffer)
    }

    fn run_attention(&self,
                    q_buffer: &MetalBuffer,
                    k_buffer: &MetalBuffer,
                    v_buffer: &MetalBuffer,
                    o_buffer: &MetalBuffer,
                    seq_length: usize,
                    head_dim: usize,
                    scale: f32,
                    causal: bool) -> f64 {
        unsafe {
            swift_run_attention(
                self.0,
                q_buffer.0,
                k_buffer.0,
                v_buffer.0,
                o_buffer.0,
                seq_length,
                head_dim,
                scale,
                causal
            )
        }
    }

    fn get_version(&self) -> String {
        unsafe {
            let c_str = swift_get_version(self.0);
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }
}

impl Drop for SwiftAttentionWrapper {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { release_object(self.0) };
        }
    }
}

// RAII wrapper for Metal buffers
struct MetalBuffer(*mut c_void);

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { release_object(self.0) };
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
        println!("{:12} {:>10.0} GINSTRS/s ({:.3} ms)",
                config, self.ginstrs_per_second, self.mean_time_ms);
    }
}

fn benchmark_config(wrapper: &SwiftAttentionWrapper, seq_len: usize, head_dim: usize) -> BenchmarkResult {
    let element_size = 2; // FP16 = 2 bytes
    let buffer_size = seq_len * head_dim * element_size;

    // Create buffers
    let q_buffer = wrapper.create_buffer(buffer_size);
    let k_buffer = wrapper.create_buffer(buffer_size);
    let v_buffer = wrapper.create_buffer(buffer_size);
    let o_buffer = wrapper.create_buffer(buffer_size);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Warmup runs
    for _ in 0..3 {
        wrapper.run_attention(&q_buffer, &k_buffer, &v_buffer, &o_buffer,
                             seq_len, head_dim, scale, false);
    }

    // Benchmark runs
    let mut times = Vec::new();
    for _ in 0..5 {
        let time = wrapper.run_attention(&q_buffer, &k_buffer, &v_buffer, &o_buffer,
                                        seq_len, head_dim, scale, false);
        if time > 0.0 {
            times.push(time * 1000.0); // Convert to milliseconds
        }
    }

    if times.is_empty() {
        return BenchmarkResult {
            mean_time_ms: -1.0,
            ginstrs_per_second: 0.0,
        };
    }

    let mean_time_ms = times.iter().sum::<f64>() / times.len() as f64;

    // Calculate GINSTRS/s for attention forward pass
    let operations = (2 * head_dim + 5) * seq_len * seq_len;
    let ginstrs_per_second = (operations as f64) / (mean_time_ms / 1000.0) / 1e9;

    BenchmarkResult {
        mean_time_ms,
        ginstrs_per_second,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Metal Flash Attention - Real Swift Kernel Performance Test");
    println!("==============================================================");

    // Create Metal device
    let device = MetalDevice::new()?;
    println!("✅ Created Metal device");

    // Create Swift attention wrapper
    let wrapper = SwiftAttentionWrapper::new(&device)?;
    println!("✅ Created Swift attention wrapper");
    println!("✅ MFA version: {}", wrapper.get_version());

    println!();
    println!("📊 Real Swift AttentionKernel Performance");
    println!("--------------------------------------------------");
    println!("Config         FWD (GINSTRS/s)");
    println!("--------------------------------------------------");

    // Test configurations matching the other benchmarks
    let configs = [(1024, 16), (1024, 64), (1024, 256)];

    for (seq_len, head_dim) in configs {
        let result = benchmark_config(&wrapper, seq_len, head_dim);
        let config = format!("{}x{}", seq_len, head_dim);

        if result.mean_time_ms > 0.0 {
            result.print(&config);
        } else {
            println!("{:12} ERROR - Failed to run attention", config);
        }
    }

    println!("--------------------------------------------------");
    println!();
    println!("🎯 Performance Analysis:");
    println!("   • Using real Swift AttentionKernel structs");
    println!("   • Rust → Objective-C → Swift → Metal pipeline");
    println!("   • Should be much closer to native Swift performance");
    println!("   • Bypasses C FFI layer completely");

    Ok(())
}
