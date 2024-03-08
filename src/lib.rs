mod texture;

use std::{fs::File, io::Read, iter::once, time::Instant};

use nalgebra::{
    ComplexField, Matrix4, Perspective3, Point3, RealField, Rotation3, Translation3, Vector3,
};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fov_y: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_projection_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(&self.eye, &self.target, &self.up);
        let projection =
            Perspective3::new(self.aspect, self.fov_y, self.znear, self.zfar).to_homogeneous();
        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

// We need this for Rust to store our data correctly for the shaders.
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    fn update_projection(&mut self, camera: &Camera) {
        self.view_proj = camera.build_projection_matrix().into();
    }
}

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_h_pressed: bool,
    is_j_pressed: bool,
    is_k_pressed: bool,
    is_l_pressed: bool,
    is_space_pressed: bool,
    is_shift_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_h_pressed: false,
            is_j_pressed: false,
            is_k_pressed: false,
            is_l_pressed: false,
            is_space_pressed: false,
            is_shift_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::H => {
                        self.is_h_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::J => {
                        self.is_j_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::K => {
                        self.is_k_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::L => {
                        self.is_l_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Space => {
                        self.is_space_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LShift => {
                        self.is_shift_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
    fn update_camera(&self, camera: &mut Camera, delta: f32) {
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();
        let angle = (camera.target.x - camera.eye.x).atan2(camera.target.z - camera.eye.z);
        let dist = 0.05 * delta;
        let dist_sin = dist * angle.sin();
        let dist_cos = dist * angle.cos();
        // Prevents glitching when the camera gets too close to the center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye.x += dist_sin;
            camera.eye.z += dist_cos;
            camera.target.x += dist_sin;
            camera.target.z += dist_cos;
        }
        if self.is_backward_pressed {
            camera.eye.x -= dist_sin;
            camera.eye.z -= dist_cos;
            camera.target.x -= dist_sin;
            camera.target.z -= dist_cos;
        }
        if self.is_right_pressed {
            camera.eye.x -= dist_cos;
            camera.eye.z += dist_sin;
            camera.target.x -= dist_cos;
            camera.target.z += dist_sin;
        }
        if self.is_left_pressed {
            camera.eye.x += dist_cos;
            camera.eye.z -= dist_sin;
            camera.target.x += dist_cos;
            camera.target.z -= dist_sin;
        }
        if self.is_h_pressed {
            camera.target = rotate_point(
                camera.eye,
                camera.target,
                -(0.5_f32.to_radians() * delta),
                'y',
            );
        }
        if self.is_j_pressed {
            //            camera.target = rotate_point(camera.eye, camera.target, 0.12_f32.to_radians(), 'x');
            if camera.target.y > camera.eye.y - 2.0 {
                camera.target.y -= 0.01 * delta;
            }
        }
        if self.is_k_pressed {
            //            camera.target = rotate_point(camera.eye, camera.target, -0.12_f32.to_radians(), 'x');
            if camera.target.y < camera.eye.y + 2.0 {
                camera.target.y += 0.01 * delta;
            }
        }
        if self.is_l_pressed {
            camera.target =
                rotate_point(camera.eye, camera.target, 0.5_f32.to_radians() * delta, 'y');
        }
        if self.is_space_pressed {
            if self.is_shift_pressed {
                camera.eye.y -= dist;
                camera.target.y -= dist;
            } else {
                camera.eye.y += dist;
                camera.target.y += dist;
            }
        }
    }
}

pub fn rotate_point(point: Point3<f32>, target: Point3<f32>, rot: f32, ax: char) -> Point3<f32> {
    let axis;
    match ax {
        'x' => axis = Vector3::x_axis(),
        'y' => axis = Vector3::y_axis(),
        'z' => axis = Vector3::z_axis(),
        _ => panic!(),
    }
    let origin_translation = Translation3::from(-point.coords);
    let rotation_matrix = Rotation3::from_axis_angle(&axis, -rot);
    let translated_point = origin_translation * target;
    let rotated_point = rotation_matrix.transform_point(&translated_point);
    let translation_back = Translation3::from(point.coords);

    translation_back * rotated_point
}

pub fn parse_obj() -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut file = File::open("models/dragon2.obj").unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for line in buffer.lines() {
        let split_line: Vec<&str> = line.split_whitespace().collect();
        if split_line[0] == "v" {
            let x: f32 = split_line[1].parse().unwrap();
            let y: f32 = split_line[2].parse().unwrap();
            let z: f32 = split_line[3].parse().unwrap();
            vertices.push([x, y, z]);
        } else if split_line[0] == "f" {
            //let mut i_list = Vec::new();
            for i in 1..4 {
                let vs: Vec<&str> = split_line[i].split('/').collect();
                let v: u32 = vs[0].parse().unwrap();
                indices.push(v - 1);
            }
            //indices.push([i_list[0], i_list[1], i_list[2]]);
        }
    }
    (vertices, indices)
}

struct State {
    /// The surface is where rendered images are presented (e.g. The part of the window which is
    /// drawn to).
    surface: wgpu::Surface,
    /// The device is a connection to a graphics (GPU) and/or
    /// compute device (CPU).
    ///
    /// The device is responsible for the creation of most rendering
    /// and compute resources.
    /// These resources are used in commands which are submitted to a `wgpu::Queue`.
    device: wgpu::Device,
    /// The queue stores and executes recorded `wgpu::CommandBuffer` objects (collections of
    /// commands).
    /// Provides convenience methods for writing to buffers and textures.
    queue: wgpu::Queue,
    /// Configuration options for a `wgpu::Surface`.
    /// In this case `State::surface`.
    config: wgpu::SurfaceConfiguration,
    /// The size of the surface display.
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    /// Winit window.
    window: Window,
    /// The Render Pipeline is a handle to a rendering (graphics) pipeline.
    /// A rendering pipeline outlines the necessary procedures for transforming a
    /// three-dimentional (3D) scene into a two-dimentional representation.
    /// In short, a rendering pipeline performs perspective projection.
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();
        // The instance is a handle to the GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window, so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::POLYGON_MODE_LINE,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web, we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            //present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: Vec::new(),
        };
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let (model_ver, model_ind) = parse_obj();
        let mut parsed_vertices = Vec::new();
        for i in model_ver {
            parsed_vertices.push(Vertex {
                position: i,
                tex_coords: [0.0; 2],
            })
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            //            contents: bytemuck::cast_slice(VERTICES),
            contents: bytemuck::cast_slice(parsed_vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            //            contents: bytemuck::cast_slice(INDICES),
            contents: bytemuck::cast_slice(model_ind.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });
        //let num_indices = INDICES.len() as u32;
        let num_indices = model_ind.len() as u32;
        surface.configure(&device, &config);
        let diffuse_bytes = include_bytes!("../happy-tree.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "../happy-tree.png")
                .unwrap();
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diffuse_bind_group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        });
        let camera = Camera {
            // positon the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: Point3::new(0.0, 2.0, 2.0),
            // have camera look at origin
            target: Point3::new(0.0, 2.0, 0.0),
            // which way is "up"
            up: Vector3::y(),
            aspect: config.width as f32 / config.height as f32,
            fov_y: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_projection(&camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Line,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let camera_controller = CameraController::new(0.0015);
        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
        }
    }
    pub fn window(&self) -> &Window {
        &self.window
    }
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }
    fn update(&mut self, delta: f32) {
        self.camera_controller
            .update_camera(&mut self.camera, delta);
        self.camera_uniform.update_projection(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // get_current_texture waits for the surface to provide a new `wgpu::SurfaceTexture` which
        // will be rendered to later.
        let output = self.surface.get_current_texture()?;
        // Create a `wgpu::Textureview` with default settings.
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Create a command encoder.
        // The command encoder creates the commands to send to the GPU. The commands are sent as a
        // command buffer.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        // Use the encoder to create a `wgpu::RenderPass`.
        // The `wgpu::RenderPass` has all the methods for the actual drawing.
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            // Describe where to draw the color to.
            // RenderPassColorAttachment has the view field, which informs wgpu what texture to
            // save the colors to.
            // Here we are using the view variable we created earlier as the view. This means any
            // colors drawn to this attachment will get drawn to the screen.
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear screen
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        /*r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,*/
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        drop(render_pass);
        // Tell wgpu to finish the command buffer and submit it to the GPU's render queue.
        // Submit will accept anything that implements IntoIter.
        self.queue.submit(once(encoder.finish()));
        output.present();
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    // Changed
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        tex_coords: [0.4131759, 0.00759614],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        tex_coords: [0.0048659444, 0.43041354],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        tex_coords: [0.28081453, 0.949397],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        tex_coords: [0.85967, 0.84732914],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        tex_coords: [0.9414737, 0.2652641],
    }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

pub async fn run() {
    // Create event loop
    let event_loop = EventLoop::new();
    // Create and initialize the window
    let window = WindowBuilder::new()
        .with_maximized(true)
        .build(&event_loop)
        .unwrap();
    let mut state = State::new(window).await;
    let test_rotation = rotate_point(
        state.camera.eye,
        state.camera.target,
        -90.0_f32.to_radians(),
        'z',
    );
    println!("{:?}", test_rotation);
    let mut frame_timer = Instant::now();
    // Initialize event loop
    event_loop.run(move |event, _, control_flow| match event {
        // Check if os has sent an event to the window
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => {
            if !state.input(event) {
                match event {
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
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            //            let now = Instant::now();
            let delta = 144.0 * frame_timer.elapsed().as_secs_f32();
            frame_timer = Instant::now();
            state.update(delta);
            //let elapsed = now.elapsed();
            //let now = Instant::now();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // Quit if system is out of memory
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
            //            println!("update: {:?}", elapsed);
            //            println!("render: {:?}", now.elapsed());
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once unless we manually request it.
            state.window.request_redraw();
        }
        _ => {}
    });
}
