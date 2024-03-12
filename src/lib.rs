mod model;
mod resources;
mod texture;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use model::Vertex;
use std::f32::consts::PI;
use std::mem::size_of_val;
use std::{fs::File, io::Read, iter::once, mem, time::Instant};

use nalgebra::{
    Matrix4, Perspective3, Point3, Rotation3, Translation3, Unit, UnitQuaternion, Vector3,
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

/// The camera is the point of view from which the scene is rendered.
pub struct Camera {
    /// The position of the camera.
    eye: Point3<f32>,
    /// The target of the camera. The target is the point the camera is looking at.
    target: Point3<f32>,
    /// The up vector of the camera. The up vector is the direction which is considered up.
    up: Vector3<f32>,
    /// The aspect ratio of the camera.
    aspect: f32,
    /// The field of view of the camera in the y direction.
    fov_y: f32,
    /// The near clipping plane of the camera.
    znear: f32,
    /// The far clipping plane of the camera.
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
/// The camera uniform is a struct which stores the projection matrix of the camera.
struct CameraUniform {
    /// The projection matrix of the camera. The projection matrix is used to transform the 3D
    /// scene into a 2D representation.
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Constructs a new `CameraUniform`.
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }
    /// Updates the projection matrix of the camera.
    fn update_projection(&mut self, camera: &Camera) {
        self.view_proj = camera.build_projection_matrix().into();
    }
}

/// The camera controller is used to control the camera's position and target.
/// The camera controller is updated based on the keyboard input.
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
    /// Constructs a new `CameraController`.
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

    /// Processes the events sent to the window. This function is used to update the state of the
    /// camera controller based on the keyboard input.
    ///
    ///
    /// # Arguments
    ///
    /// * `event`:
    ///
    /// returns: bool
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
    /// Updates the camera's position and target based on the keyboard input.
    fn update_camera(&self, camera: &mut Camera, delta: f32) {
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();
        let angle = (camera.target.x - camera.eye.x).atan2(camera.target.z - camera.eye.z);
        let dist = 0.065 * delta;
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
                camera.target.y -= 0.015 * delta;
            }
        }
        if self.is_k_pressed {
            //            camera.target = rotate_point(camera.eye, camera.target, -0.12_f32.to_radians(), 'x');
            if camera.target.y < camera.eye.y + 2.0 {
                camera.target.y += 0.015 * delta;
            }
        }
        if self.is_l_pressed {
            camera.target =
                rotate_point(camera.eye, camera.target, 0.5_f32.to_radians() * delta, 'y');
        }
        if self.is_space_pressed {
            if self.is_shift_pressed {
                camera.eye.y -= dist * 1.5;
                camera.target.y -= dist * 1.5;
            } else {
                camera.eye.y += dist * 1.5;
                camera.target.y += dist * 1.5;
            }
        }
    }
}

/// Rotates a point around a target point. The rotation is done in 3D space. The rotation is
/// performed around the specified axis.
///
///
/// # Arguments
///
/// * `point`:
/// * `target`:
/// * `rot`:
/// * `ax`:
///
/// returns: OPoint<f32, Const<3>>

// This is so we can store this in a buffer
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransformationMatrix {
    tmatrix: [[f32; 4]; 4],
    origin_translation: [f32; 3],
    translation_back: [f32; 3],
    _padding: [f32; 2],
}

impl TransformationMatrix {
    fn new() -> Self {
        rotate_points_amount(Point3::default(), 0.0, 'y')
    }
}

pub fn rotate_point(point: Point3<f32>, target: Point3<f32>, rot: f32, ax: char) -> Point3<f32> {
    let axis = match ax {
        'x' => Vector3::x_axis(),
        'y' => Vector3::y_axis(),
        'z' => Vector3::z_axis(),
        _ => panic!(),
    };
    let origin_translation = Translation3::from(-point.coords);
    let rotation_matrix = Rotation3::from_axis_angle(&axis, -rot);
    let translated_point = origin_translation * target;
    let rotated_point = rotation_matrix.transform_point(&translated_point);
    let translation_back = Translation3::from(point.coords);

    translation_back * rotated_point
}

/// Rotates a list of points around a target point. The rotation is done in 3D space. The rotation
/// is performed around the specified axis.
///
///
/// # Arguments
///
/// * `point`:
/// * `targets`:
/// * `rot`:
/// * `ax`:
///
/// returns: Vec<Vertex, Global>
/*
fn rotate_points(point: Point3<f32>, targets: &Vec<Vertex>, rot: f32, ax: char) -> Vec<Vertex> {
    let axis;
    match ax {
        'x' => axis = Vector3::x_axis(),
        'y' => axis = Vector3::y_axis(),
        'z' => axis = Vector3::z_axis(),
        _ => panic!(),
    }
    let origin_translation = Translation3::from(-point.coords);
    //let now = Instant::now();
    let rotation_matrix = Rotation3::from_axis_angle(&axis, -rot);
    let q = UnitQuaternion::from_rotation_matrix(&rotation_matrix);
    let translation_back = Translation3::from(point.coords);
    let r_points = targets
        .par_iter()
        .map(|t| {
            if t.position[1] == 0.0 {
                return *t;
            }
            let translated_point = origin_translation * Point3::from(t.position);
            //let rotated_point = rotation_matrix.transform_point(&translated_point);
            let rotated_point = q.transform_point(&translated_point);
            let pos = (translation_back * rotated_point).coords;
            Vertex {
                position: [pos.x, pos.y, pos.z],
                tex_coords: t.tex_coords,
            }
        })
        .collect();
    //let el = now.elapsed();
    //println!("{:?}", el);
    r_points
}
*/
fn rotate_points_amount(point: Point3<f32>, rot: f32, ax: char) -> TransformationMatrix {
    let axis = match ax {
        'x' => Vector3::x_axis(),
        'y' => Vector3::y_axis(),
        'z' => Vector3::z_axis(),
        _ => panic!(),
    };
    let origin_translation = Translation3::from(-point.coords);
    //let now = Instant::now();
    let rotation_matrix = Rotation3::from_axis_angle(&axis, -rot);
    //let q = UnitQuaternion::from_rotation_matrix(&rotation_matrix);
    let translation_back = Translation3::from(point.coords);
    TransformationMatrix {
        tmatrix: rotation_matrix.to_homogeneous().into(),
        origin_translation: origin_translation.into(),
        translation_back: translation_back.into(),
        _padding: [0.0, 0.0],
    }
    /*
    let r_points = targets
        .par_iter()
        .map(|t| {
            if t.position[1] == 0.0 {
                return *t;
            }
            let translated_point = origin_translation * Point3::from(t.position);
            //let rotated_point = rotation_matrix.transform_point(&translated_point);
            let rotated_point = q.transform_point(&translated_point);
            let pos = (translation_back * rotated_point).coords;
            Vertex {
                position: [pos.x - t.position[0], pos.y - t.position[1], pos.z - t.position[2]],
                tex_coords: t.tex_coords,
            }
        })
        .collect();
    //let el = now.elapsed();
    //println!("{:?}", el);
    let mut conv: Vec<[f32; 3]> = Vec::new();
    for i in r_points {
        conv.push(i.positon);
    }*/
}

/// Parses the obj file and returns the vertices and indices.
///
///
/// # Arguments
///
///
///
/// returns: (Vec<[f32; 3]>, Vec<u32>)
pub fn parse_obj(file: &str) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut file = File::open(file).unwrap();
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

struct Instance {
    position: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: ((Translation3::from(self.position).to_homogeneous())
                * self.rotation.to_homogeneous())
            .into(),
        }
    }
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to
                // define a slot for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 1;
const INSTANCE_DISPLACEMENT: Vector3<f32> = Vector3::new(
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
    0.0,
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
);

const INSTANCE_SPACING: f32 = 9.0;

/// The state of the program. This is where the main logic of the program is stored.
/// This is the first struct that is created when the program is run.
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
    render_pipeline: wgpu::RenderPipeline,
    /// The vertex buffer is a handle to a buffer which stores vertex data.
    /// Vertex data is used to describe the shape of a 3D object by defining points (vertices).
    /// The index buffer stores the conncections between vertices.
    /// The index buffer is a handle to a buffer which stores index data.
    /// Index data is used to describe the connections between vertices.
    /// The diffuse bind group is a handle to a bind group which stores the diffuse texture.
    /// A bind group is a collection of resources which are bound to the pipeline. These resources
    /// are used in the shaders.
    diffuse_bind_group: wgpu::BindGroup,
    /// The diffuse texture is a handle to a texture which stores the color data of the diffuse
    /// texture.
    //diffuse_texture: texture::Texture,
    /// The camera is the point of view from which the scene is rendered.
    camera: Camera,
    /// The camera uniform is a struct which stores the camera's view and projection.
    camera_uniform: CameraUniform,
    /// The camera buffer is a handle to a buffer which stores the camera's view and projection
    /// data.
    /// The camera buffer is used to update the camera's view and projection in the shaders.
    camera_buffer: wgpu::Buffer,
    /// The camera bind group is a handle to a bind group which stores the camera buffer.
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,
    obj_model: model::Model,
}

impl State {
    /// Constructs a new `State`.
    ///
    ///
    /// # Arguments
    ///
    /// * `window`:
    ///
    /// returns: State
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
                    features: wgpu::Features::POLYGON_MODE_LINE
                        | wgpu::Features::CONSERVATIVE_RASTERIZATION,
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
            present_mode: surface_caps.present_modes[0],
            //present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: Vec::new(),
        };
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        surface.configure(&device, &config);
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let pos = Vector3::new(
                        x as f32 * INSTANCE_SPACING,
                        0.0,
                        z as f32 * INSTANCE_SPACING,
                    ) - INSTANCE_DISPLACEMENT;
                    let rot = if pos.norm_squared() == 0.0 {
                        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0)
                    } else {
                        UnitQuaternion::from_axis_angle(
                            &Unit::new_normalize(pos),
                            0.0_f32.to_radians(),
                        )
                    };
                    Instance {
                        position: pos,
                        rotation: rot,
                    }
                })
            })
            .collect::<Vec<_>>();
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let transform_uniform = TransformationMatrix::new();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&[transform_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("transform_bind_group_layout"),
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
        println!("{:?}", size_of_val(&transform_uniform));
        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bind_group"),
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
        });
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
        let obj_model =
            resources::load_model("dragon6.obj", &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();
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
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &transform_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
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
                //cull_mode: None,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
            diffuse_bind_group,
            //            diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            instances,
            instance_buffer,
            transform_buffer,
            transform_bind_group,
            depth_texture,
            obj_model,
        }
    }
    pub fn window(&self) -> &Window {
        &self.window
    }
    ///
    ///
    /// # Arguments
    ///
    /// * `new_size`:
    ///
    /// returns: ()
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }
    ///
    ///
    /// # Arguments
    ///
    /// * `event`:
    ///
    /// returns: bool
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }
    ///
    ///
    /// # Arguments
    ///
    /// * `delta`:
    ///
    /// returns: ()
    fn update(&mut self, delta: f32, rot: f32) {
        self.camera_controller
            .update_camera(&mut self.camera, delta);
        self.camera_uniform.update_projection(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[rotate_points_amount(Point3::default(), rot, 'y')]),
        );
    }
    ///
    ///
    /// # Arguments
    ///
    /// returns: Result<(), wgpu::SurfaceError>
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // get_current_texture waits for the surface to provide a new `wgpu::SurfaceTexture` which
        // will be rendered to later.
        let now = Instant::now();
        let output = self.surface.get_current_texture()?;
        // Create a `wgpu::Textureview` with default settings.
        println!("draw: {:?}", now.elapsed());
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
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.transform_bind_group, &[]);
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for i in 0..self.obj_model.meshes.len() {
            let mesh = &self.obj_model.meshes[i];
            let material = &self.obj_model.materials[mesh.material];
            use model::DrawModel;
            render_pass.draw_mesh_instanced(
                mesh,
                material,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
            );
        }
        drop(render_pass);
        // Tell wgpu to finish the command buffer and submit it to the GPU's render queue.
        // Submit will accept anything that implements IntoIter.
        self.queue.submit(once(encoder.finish()));
        output.present();
        Ok(())
    }
}

/// This is the entry point of the program.
#[macro_use]
extern crate log;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
    } else {
        env_logger::init();
    }
    }
    // Create event loop
    let event_loop = EventLoop::new();
    // Create and initialize the window
    let window = WindowBuilder::new()
        .with_maximized(true)
        .build(&event_loop)
        .unwrap();
    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = State::new(window).await;
    let test_rotation = rotate_point(
        state.camera.eye,
        state.camera.target,
        -90.0_f32.to_radians(),
        'z',
    );
    println!("{:?}", test_rotation);
    let mut switch = false;
    let mut frame_timer = Instant::now();
    let mut rot = 0.0;
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
            let el = frame_timer.elapsed();
            let delta = 144.0 * el.as_secs_f32();
            if switch {
                println!("{:?}fps", (144.0 / delta) as u32);
            }
            switch = !switch;
            frame_timer = Instant::now();
            rot += 0.001 * delta;
            if rot >= PI * 2.0 {
                rot = 0.0;
            }
            state.update(delta, rot);
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // Quit if system is out of memory
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once unless we manually request it.
            state.window.request_redraw();
        }
        _ => {}
    });
}
