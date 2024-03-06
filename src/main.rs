use wgpu_tutorial1::run;

fn main() {
    // Initialize graphics
    pollster::block_on(run());
}
