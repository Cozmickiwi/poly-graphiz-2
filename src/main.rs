use wgpu_tutorial1::run;

#[macro_use]
extern crate log;

fn main() {
    // Enable error logging
    env_logger::init();
    // Initialize graphics
    pollster::block_on(run());
}
