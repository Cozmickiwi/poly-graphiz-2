use std::io::Cursor;

use gltf::image::Format;
use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use nalgebra::{Point3, Rotation3, Vector3};
use wgpu::util::DeviceExt;

use crate::{model, texture};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    address_mode: wgpu::AddressMode,
    transform: [f32; 3],
    rotation: f32,
    tmindex: u32,
) -> anyhow::Result<model::Model> {
    return Ok(load_glb(
        device,
        queue,
        layout,
        file_name,
        address_mode,
        transform,
        rotation,
        tmindex,
    ));
}
fn load_glb(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    file_name: &str,
    address_mode: wgpu::AddressMode,
    transform: [f32; 3],
    rotation: f32,
    tmindex: u32,
) -> model::Model {
    let (gltf, buffers, imgs) = gltf::import(file_name).unwrap();
    let mut mesh_count: u32 = 0;
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    //let angle = -std::f32::consts::FRAC_PI_2;
    //let angle = 0.0;
    let axis = Vector3::x_axis();
    let rot = Rotation3::from_axis_angle(&axis, rotation);
    println!("Images: {}", imgs.len());
    for mesh in gltf.meshes() {
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut point3v = Vec::new();
        let mut vertices2 = Vec::new();
        let mut indices = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut texture_coords: Vec<[f32; 2]> = Vec::new();
        let mut images = Vec::new();
        for i in &imgs {
            if i.format != Format::R8G8 && i.format != Format::R8 {
                images.push(i);
            }
        }
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            if let Some(viter) = reader.read_positions() {
                vertices = viter.collect();
            }
            for v in &vertices {
                let mut point = Point3::new(v[0], v[1], v[2]);
                point = rot.transform_point(&point);
                point3v.push([
                    point.x + transform[0],
                    point.y + transform[1],
                    point.z + transform[2],
                ]);
            }
            if let Some(normiter) = reader.read_normals() {
                normals = normiter.collect();
            }
            if let Some(texiter) = reader.read_tex_coords(0) {
                texture_coords = texiter.into_f32().collect();
            }
            for i in 0..vertices.len() {
                vertices2.push(model::ModelVertex {
                    position: point3v[i],
                    tex_coords: texture_coords[i],
                    //tex_coords: [texture_coords[i][0] / 4.0, texture_coords[i][1] / 4.0],
                    normal: normals[i],
                    tmindex,
                });
            }
            if let Some(initer) = reader.read_indices() {
                indices = initer.into_u32().collect();
            }
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", mesh_count)),
            contents: bytemuck::cast_slice(&vertices2),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", mesh_count)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        meshes.push(model::Mesh {
            name: format!("Mesh{mesh_count}"),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: mesh_count as usize,
        });
        let img;
        if (mesh_count as usize) < images.len() {
            img = &images[mesh_count as usize];
        } else {
            img = &images[0];
        }
        let dyn_image: DynamicImage;
        if img.format == Format::R8G8B8 || img.format == Format::R8G8 {
            let image_buffer = ImageBuffer::<Rgb<u8>, _>::from_fn(img.width, img.height, |x, y| {
                if img.format == Format::R8G8 {
                    let index = (y * img.width + x) as usize * 2;
                    if index >= img.pixels.len() - 2 {
                        return Rgb([
                            img.pixels[img.pixels.len() - 4],
                            img.pixels[img.pixels.len() - 3],
                            0,
                        ]);
                    }
                    return Rgb([img.pixels[index], img.pixels[index + 1], 255]);
                } else {
                    let index = (y * img.width + x) as usize * 3;
                    if index >= img.pixels.len() - 3 {
                        return Rgb([
                            img.pixels[img.pixels.len() - 4],
                            img.pixels[img.pixels.len() - 3],
                            img.pixels[img.pixels.len() - 2],
                        ]);
                    }
                    return Rgb([
                        img.pixels[index],
                        img.pixels[index + 1],
                        img.pixels[index + 2],
                    ]);
                }
            });
            dyn_image = DynamicImage::ImageRgb8(image_buffer);
        } else if img.format == Format::R8G8B8A8 {
            let image_buffer =
                ImageBuffer::<Rgba<u8>, _>::from_fn(img.width, img.height, |x, y| {
                    let index = (y * img.width + x) as usize * 4;
                    if index >= img.pixels.len() - 4 {
                        return Rgba([
                            img.pixels[img.pixels.len() - 4],
                            img.pixels[img.pixels.len() - 3],
                            img.pixels[img.pixels.len() - 2],
                            img.pixels[img.pixels.len() - 1],
                        ]);
                    }
                    Rgba([
                        img.pixels[index],
                        img.pixels[index + 1],
                        img.pixels[index + 2],
                        img.pixels[index + 3],
                    ])
                });
            dyn_image = DynamicImage::ImageRgba8(image_buffer);
        } else {
            panic!("Unsupported image format: {:?}", img.format);
        }
        let mut buffer: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        dyn_image.resize(
            dyn_image.width() / 10,
            dyn_image.height() / 10,
            image::imageops::FilterType::Lanczos3,
        );
        dyn_image
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        let diffuse_texture = texture::Texture::from_bytes(
            device,
            queue,
            bytemuck::cast_slice(cursor.get_ref()),
            &format!("Mesh{mesh_count}"),
            address_mode,
        )
        .unwrap();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
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
            label: None,
        });
        materials.push(model::Material {
            name: format!("Mesh{mesh_count}"),
            diffuse_texture,
            bind_group,
        });
        mesh_count += 1;
    }
    println!("meshes: {}", meshes.len());
    model::Model {
        meshes,
        materials,
        pos: transform,
    }
}
