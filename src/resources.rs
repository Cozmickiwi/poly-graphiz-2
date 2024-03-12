use std::env::{current_exe, set_current_dir};
use std::fs::{read_dir, File};
use std::io::{BufReader, Cursor, Read};

use cfg_if::cfg_if;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};

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

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;
        } else {
        /*
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
        */  let txt = std::fs::read_to_string("res/girl3.obj")?;
        }
    }

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let url = format_url(file_name);
                let data = reqwest::get(url)
                    .await?
                    .bytes()
                    .await?
                    .to_vec();
            } else {
    /*
                let path = std::path::Path::new(env!("OUT_DIR"))
                    .join("res")
                    .join(file_name);*/
                //let data = std::fs::read(path)?;
                let data = std::fs::read("res/girl3.obj")?;
            }
        }

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &Device,
    queue: &Queue,
) -> anyhow::Result<texture::Texture> {
//    let data = load_binary(file_name).await?;
    let data = std::fs::read("res/girl3.mtl")?;
    texture::Texture::from_bytes(&device, &queue, &data, file_name)
}

pub async fn load_model(
    file_name: &str,
    device: &Device,
    queue: &Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    /*
    for i in read_dir("target/release").unwrap() {
        println!("{:?}", i);
    }*/
    let obj_text = load_string(file_name).await?;
    //let mut file = File::open("dragon5.obj").expect("E1");
    //    let mut obj_text = String::new();
    //    file.read_to_string(&mut obj_text);
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);
    let obj_materials = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            //let mat_text = load_string("girl3.mtl").await.unwrap();
        //    println!("{mat_text}");
            //return tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
            let mut file = File::open("res/girl3.obj").unwrap();
            return tobj::load_mtl_buf(&mut BufReader::new(file))
        },
    )
    .await?;
    let mut materials = Vec::new();
    //println!("{:?}", obj_materials.1.unwrap().len());
    for m in obj_materials.1.unwrap() {
        //println!("tex: {:?}", &m.dissolve_texture);
        let mut file = File::open("res/girl3.mtl").unwrap();
        let mut buf = String::new();
        let diffuse_texture = load_texture("girl3.mtl", device, queue)
            .await
            .expect("E3");
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
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }
    let meshes = obj_materials.0
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();
    println!("{:?}", materials.len());
    Ok(model::Model { meshes, materials })
}
