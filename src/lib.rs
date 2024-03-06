use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: Window,
}

impl State<'_> {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        todo!()
    }
    pub fn window(&self) -> &Window {
        &self.window
    }
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        todo!()
    }
    fn input(&mut self, event: &WindowEvent) -> bool {
        todo!()
    }
    fn update(&mut self) {
        todo!()
    }
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        todo!()
    }
}

pub fn run() {
    // Enable error logging
    env_logger::init();
    // Create event loop
    let event_loop = EventLoop::new();
    // Create and initialize the window
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    // Initialize event loop
    event_loop.run(move |event, _, control_flow| match event {
        // Check if os has sent an event to the window
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            // Close the window when close has been requested (e.g window close button pressed) or escape is pressed
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        },
        _ => {}
    });
}
