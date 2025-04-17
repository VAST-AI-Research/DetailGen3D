import argparse
import os
import torch
import trimesh
import numpy as np
from PIL import Image
from skimage import measure
from huggingface_hub import snapshot_download

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from detailgen3d.inference_utils import generate_dense_grid_points
from detailgen3d.pipelines.pipeline_detailgen3d import (
    DetailGen3DPipeline,
)

def load_mesh(mesh_path, num_pc=20480):
    mesh = trimesh.load(mesh_path,force="mesh")

    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(1.9 / scale)

    surface, face_indices = trimesh.sample.sample_surface(mesh, 1000000,)
    normal = mesh.face_normals[face_indices]

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()

    return surface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_input", type=str, default="assets/model/503d193a-1b9b-4685-b05f-00ac82f93d7b.glb")
    parser.add_argument("--image_input", type=str, default="assets/image/503d193a-1b9b-4685-b05f-00ac82f93d7b.png")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_aug", type=float, default=0)
    parser.add_argument("--num-inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # prepare pipeline
    local_dir = "pretrained_weights/DetailGen3D"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id="VAST-AI/DetailGen3D", local_dir=local_dir)
    pipeline = DetailGen3DPipeline.from_pretrained(
        local_dir
    ).to(device, dtype=dtype)

    # prepare data
    image_path = args.image_input
    image = Image.open(image_path).convert("RGB")

    mesh_path = args.mesh_input
    surface = load_mesh(mesh_path).to(device, dtype=dtype)

    batch_size = 1

    # sample query points for decoding
    box_min = np.array([-1.005, -1.005, -1.005])
    box_max = np.array([1.005, 1.005, 1.005])
    sampled_points, grid_size, bbox_size = generate_dense_grid_points(
        bbox_min=box_min, bbox_max=box_max, octree_depth=9, indexing="ij"
    )
    sampled_points = torch.FloatTensor(sampled_points).to(device, dtype=dtype)
    sampled_points = sampled_points.unsqueeze(0).repeat(batch_size, 1, 1)

    # inference pipeline
    sample = pipeline.vae.encode(surface).latent_dist.sample()
    sdf = pipeline(
        image, 
        latents=sample, 
        sampled_points=sampled_points, 
        noise_aug_level=args.noise_aug, 
        generator=torch.Generator(device=device).manual_seed(args.seed),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    ).samples[0]

    # marching cubes
    grid_logits = sdf.view(grid_size).cpu().numpy()
    vertices, faces, normals, _ = measure.marching_cubes(
        grid_logits, 0, method="lewiner"
    )
    vertices = vertices / grid_size * bbox_size + box_min
    mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, args.mesh_input.split("/")[-1].split(".")[0]+".glb")
    mesh.export(output_dir, file_type="glb")
