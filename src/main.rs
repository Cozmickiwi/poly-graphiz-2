use wgpu_tutorial1::run;

fn main() {
    // Enable error logging
    env_logger::init();
    // Initialize graphics
    pollster::block_on(run());
}
