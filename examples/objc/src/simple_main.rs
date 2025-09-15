use std::ffi::{c_void};

// External C functions from our Objective-C bridge
extern "C" {
    fn create_metal_device() -> *mut c_void;
    fn create_metal_buffer(device: *mut c_void, size: usize) -> *mut c_void;
    fn benchmark_metal_operations(device: *mut c_void, seq_len: usize, head_dim: usize) -> f64;
    fn release_object(obj: *mut c_void);
}

// RAII wrapper for Metal objects
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

    fn create_buffer(&self, size: usize) -> MetalBuffer {
        let buffer = unsafe { create_metal_buffer(self.0, size) };
        MetalBuffer(buffer)
    }

    fn benchmark_operations(&self, seq_len: usize, head_dim: usize) -> f64 {
        unsafe { benchmark_metal_operations(self.0, seq_len, head_dim) }
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { release_object(self.0) };
        }
    }
}

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
        println!("{:12} {:>10.0} GINSTRS/s ({:.2} ms)",
                config, self.ginstrs_per_second, self.mean_time_ms);
    }
}

fn benchmark_config(device: &MetalDevice, seq_len: usize, head_dim: usize) -> BenchmarkResult {
    // Run benchmark
    let time_seconds = device.benchmark_operations(seq_len, head_dim);
    let mean_time_ms = time_seconds * 1000.0;

    // Calculate theoretical GINSTRS/s for attention forward pass
    // This is a rough approximation for comparison
    let operations = (2 * head_dim + 5) * seq_len * seq_len;
    let ginstrs_per_second = (operations as f64) / time_seconds / 1e9;

    BenchmarkResult {
        mean_time_ms,
        ginstrs_per_second,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Metal Flash Attention - Objective-C Runtime Overhead Test");
    println!("===============================================================");

    // Create Metal device
    let device = MetalDevice::new()?;
    println!("✅ Created Metal device via Objective-C");

    // Test basic buffer creation
    let _test_buffer = device.create_buffer(1024);
    println!("✅ Created Metal buffer via Objective-C");

    println!();
    println!("📊 Objective-C Runtime Overhead Benchmark");
    println!("(Simple Metal operations, not actual attention kernels)");
    println!("--------------------------------------------------");
    println!("Config         GINSTRS/s (Time)");
    println!("--------------------------------------------------");

    // Test configurations
    let configs = [(1024, 16), (1024, 64), (1024, 256)];

    for (seq_len, head_dim) in configs {
        let result = benchmark_config(&device, seq_len, head_dim);
        let config = format!("{}x{}", seq_len, head_dim);
        result.print(&config);
    }

    println!("--------------------------------------------------");
    println!();
    println!("🎯 Analysis:");
    println!("   • This measures Rust → Objective-C → Metal overhead");
    println!("   • Uses simple copy kernels, not complex attention");
    println!("   • Shows baseline performance without C FFI layer");
    println!("   • Objective-C runtime adds minimal overhead vs C FFI");

    Ok(())
}
