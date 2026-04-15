import os.path

from PIL import Image
from torchvision import transforms
from src.general_utils import load_video, rank0_log, get_last_video_frame
import torch
import torch.nn.functional as F
import einops
import json
import numpy as np
from models.camera import get_camera_embedding, camera_center_normalization, matrix_to_quaternion, unified_camera_normalization
from glob import glob
from src.pointcloud import get_points3d_and_colors, multi_gpu_point_rendering
from diffusers.utils import export_to_video
import torch.distributed as dist


def assign_scale(h, w, scale_map=None):
    if scale_map is None:
        scale_map = [[480, 768], [512, 720], [576, 640], [608, 608], [640, 576], [720, 512], [768, 480]]
    hw_ratio = h / w
    map_ratios = [h_ / w_ for h_, w_ in scale_map]
    best_scale_idx = int(np.argmin(np.abs(np.array(map_ratios) - hw_ratio)))
    best_scale = scale_map[best_scale_idx]

    return best_scale


def sort_trajs(scene_path):
    # Generation order: (view*: up --> right --> left) --> target* --> reconstruct* --> wonder*
    view_paths = glob(f"{scene_path}/view*")
    view_paths.sort()
    view_list = []
    for view_path in view_paths:
        if os.path.exists(f"{view_path}/valid_analysis.json"):
            v = json.load(open(f"{view_path}/valid_analysis.json"))
            if v["valid"] != "yes":  # skip invalid views
                continue
        traj_paths_ = glob(f"{view_path}/traj*/render.mp4")
        traj_paths_.sort()
        traj_paths = []
        if f"{view_path}/traj2/render.mp4" in traj_paths_:
            traj_paths.append(f"{view_path}/traj2/render.mp4")
        if f"{view_path}/traj0/render.mp4" in traj_paths_:
            traj_paths.append(f"{view_path}/traj0/render.mp4")
        if f"{view_path}/traj1/render.mp4" in traj_paths_:
            traj_paths.append(f"{view_path}/traj1/render.mp4")
        view_list.extend(traj_paths)
    target_list = glob(f"{scene_path}/target*/traj*/render.mp4")
    target_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    recon_list = glob(f"{scene_path}/reconstruct*/traj0/render.mp4")
    recon_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    wander_list = glob(f"{scene_path}/wonder*/traj*/render.mp4")
    wander_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    total_list = view_list + target_list + recon_list + wander_list

    # If iterative trajectories (reconstruct_*/traj1) exist, place them at the end
    iter_list = glob(f"{scene_path}/reconstruct*/traj1/render.mp4")
    iter_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    total_list = total_list + iter_list

    # Rename any paths whose 3rd-to-last directory contains "-" by replacing it with "_"
    total_list = rename_hyphen_to_underscore(total_list)

    return total_list


def recon_sort_trajs(scene_path):
    # Generation order: eloop --> up --> right --> left
    total_list = [f"{scene_path}/traj_eloop/render.mp4",
                  f"{scene_path}/traj_up/render.mp4",
                  f"{scene_path}/traj_right/render.mp4",
                  f"{scene_path}/traj_left/render.mp4"]

    return total_list


def rename_hyphen_to_underscore(path_list):
    """
    Check whether the 3rd-to-last directory component (i.e. the view*/target*/reconstruct*/wonder* level)
    contains a hyphen. If so, rename that directory by replacing '-' with '_' and update the path list.
    """
    new_path_list = []
    for old_path in path_list:
        parts = old_path.split('/')
        dir_name = parts[-3]  # 3rd-to-last directory name
        if '-' not in dir_name:
            new_path_list.append(old_path)
            continue

        new_dir_name = dir_name.replace('-', '_')
        # Build old and new directory paths
        parent_path = '/'.join(parts[:-3])
        old_dir_path = os.path.join(parent_path, dir_name) if parent_path else dir_name
        new_dir_path = os.path.join(parent_path, new_dir_name) if parent_path else new_dir_name
        if os.path.exists(old_dir_path) and not os.path.exists(new_dir_path):
            os.rename(old_dir_path, new_dir_path)

        # Update directory name in path
        parts[-3] = new_dir_name
        new_path_list.append('/'.join(parts))
    return new_path_list


def load_mutli_traj_dataset(cfg, input_path, output_path, view_id, traj_id, device, ref_index=None, model_type=None, task_type="panorama"):
    if view_id.startswith("reconstruct_") and traj_id == "traj1":  # iterative traj
        pil_image = get_last_video_frame(f"{output_path}/{view_id}/traj0/{model_type}_result.mp4")
        pil_image = Image.fromarray(pil_image)
    else:
        if task_type == "panorama":
            pil_image = Image.open(f"{input_path}/{view_id}/start_frame.png")
        else:
            pil_image = Image.open(f"{input_path}/image.png")
    origin_w, origin_h = pil_image.size
    height, width = assign_scale(origin_h, origin_w, scale_map=cfg.scale_map)
    pil_image = pil_image.resize((width, height), Image.Resampling.BICUBIC)
    meta = {"image": pil_image}

    image = transforms.ToTensor()(pil_image) * 2 - 1

    # load conditions
    render_frames = load_video(f"{input_path}/{view_id}/{traj_id}/render.mp4")
    render_video = torch.stack([transforms.ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
    # replace the first frame with the high-quality reference image
    render_video[0, 0] = image
    render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
    render_mask_frames = load_video(f"{input_path}/{view_id}/{traj_id}/render_mask.mp4")
    render_mask = torch.stack([transforms.ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
    render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]  # [0,1],[1,f,1,h,w]
    if view_id.startswith("reconstruct_") and traj_id == "traj1":  # iterative traj
        render_mask[0, 0] = 0
    render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1
    meta["render_video"] = render_video.to(device)
    meta["render_mask"] = render_mask.to(device)

    if meta["render_video"].shape[2] > cfg.nframe:
        indices = np.linspace(0, meta["render_video"].shape[2] - 1, cfg.nframe, dtype=int)
        meta["render_video"] = meta["render_video"][:, :, indices]
    if meta["render_mask"].shape[2] > cfg.nframe:
        indices = np.linspace(0, meta["render_mask"].shape[2] - 1, cfg.nframe, dtype=int)
        meta["render_mask"] = meta["render_mask"][:, :, indices]

    # load camera
    cam_info = json.load(open(f"{input_path}/{view_id}/{traj_id}/camera.json"))
    w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
    intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)
    intrinsic[:, 0, :] = intrinsic[:, 0, :] / origin_w * width
    intrinsic[:, 1, :] = intrinsic[:, 1, :] / origin_h * height

    if w2cs.shape[0] > cfg.nframe:
        indices = np.linspace(0, w2cs.shape[0] - 1, cfg.nframe, dtype=int)
        w2cs = w2cs[indices]
    if intrinsic.shape[0] > cfg.nframe:
        indices = np.linspace(0, intrinsic.shape[0] - 1, cfg.nframe, dtype=int)
        intrinsic = intrinsic[indices]

    camera_embedding = get_camera_embedding(intrinsic, w2cs, intrinsic.shape[0], height, width, normalize=True, is_w2c=True)

    if os.path.exists(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}_ref_w2cs.json"):
        extrinsic_ref = json.load(open(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}_ref_w2cs.json"))
        extrinsic_ref = torch.tensor(np.array(extrinsic_ref), dtype=torch.float32, device=device)

        # Unified normalization: coordinate system reset based on all cameras,
        extrinsic, extrinsic_ref = unified_camera_normalization(w2cs, extrinsic_ref)

        # Convert R matrix (w2c) to quaternion and concat with t vector
        R_matrix = extrinsic[:, :3, :3]
        t_vector = extrinsic[:, :3, 3]
        quaternion = matrix_to_quaternion(R_matrix)
        camera_qt = torch.cat([quaternion, t_vector], dim=-1).unsqueeze(0)  # [1, f, 7]

        R_matrix_ref = extrinsic_ref[:, :3, :3]
        t_vector_ref = extrinsic_ref[:, :3, 3]
        quaternion_ref = matrix_to_quaternion(R_matrix_ref)
        camera_qt_ref = torch.cat([quaternion_ref, t_vector_ref], dim=-1).unsqueeze(0)  # [1, f_ref, 7]

        meta["camera_qt"] = camera_qt.to(device)
        meta["camera_qt_ref"] = camera_qt_ref.to(device)

    meta["camera_embedding"] = camera_embedding.to(device)
    meta["extrinsics"] = w2cs.to(device)
    meta["intrinsics"] = intrinsic.to(device)

    # load reference
    combined_frames = load_video(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}.mp4")
    combined_frames = [np.array(frame) for frame in combined_frames]
    reference_video = torch.stack([transforms.ToTensor()(frame[:, :]) for frame in combined_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    reference_video = F.interpolate(reference_video, size=(height, width), mode='bicubic')

    # Filter frames based on ref_index if provided
    if ref_index is not None and len(ref_index) > 0:
        reference_video = reference_video[ref_index]

    meta["reference_video"] = einops.rearrange(reference_video[None], "b f c h w -> b c f h w").to(device)
    meta["ref_index"] = ref_index

    # load text
    if task_type == "panorama":
        prompt = json.load(open(f"{input_path}/{view_id}/{traj_id}/traj_caption.json"))["prompt"]
    else:
        prompt = json.load(open(f"{input_path}/prompt.json"))["prompt"]
    meta["prompt"] = prompt

    meta["width"] = width
    meta["height"] = height

    return meta


def load_single_view_data(cfg, input_path, output_path, model_type, depth_model, device, sp_size=1, sp_rank=0):
    pil_image = Image.open(f"{input_path}/image.png")
    origin_w, origin_h = pil_image.size

    height, width = assign_scale(origin_h, origin_w, scale_map=cfg.scale_map)

    rank0_log(f"Resized image to {height}x{width}")
    pil_image = pil_image.resize((width, height), Image.Resampling.BICUBIC)
    image = transforms.ToTensor()(pil_image) * 2 - 1

    # load camera
    cam_info = json.load(open(f"{input_path}/camera.json"))
    w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
    intrinsics = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)

    if w2cs.shape[0] > cfg.nframe:
        indices = np.linspace(0, w2cs.shape[0] - 1, cfg.nframe, dtype=int)
        w2cs = w2cs[indices]
    if intrinsics.shape[0] > cfg.nframe:
        indices = np.linspace(0, intrinsics.shape[0] - 1, cfg.nframe, dtype=int)
        intrinsics = intrinsics[indices]

    intrinsics[:, 0, :] = intrinsics[:, 0, :] / origin_w * width
    intrinsics[:, 1, :] = intrinsics[:, 1, :] / origin_h * height

    # load conditions
    # predict depth via moge
    depth_pred = depth_model.infer(transforms.ToTensor()(pil_image)[None].to(device))
    depth_mask = depth_pred["mask"][0]
    depth = depth_pred["depth"][0].clone()
    depth[~depth_mask] = 0

    points_3d, points_color = get_points3d_and_colors(K=intrinsics, w2cs=w2cs, depth=depth[None, None],
                                                      image=image[None], device=device, mask=~depth_mask)
    pcd_renders, pcd_mask = multi_gpu_point_rendering(image=pil_image, Ks=intrinsics, w2cs=w2cs,
                                                      render_points=points_3d,
                                                      render_colors=points_color,  # [-1~1]
                                                      image_h=height, image_w=width,
                                                      device=device, device_num=sp_size,
                                                      render_radius=0.008, points_per_pixel=20,
                                                      slice_size=4, local_rank=sp_rank, replace_first_frame=True)

    render_video = []
    mask_video = []
    pcd_renders = pcd_renders.to(torch.float32)
    for j in range(pcd_renders.shape[0]):
        img = transforms.ToPILImage()((pcd_renders[j] + 1) / 2)
        render_video.append(img)
        mask = transforms.ToPILImage()(pcd_mask[j])
        mask_video.append(mask)

    if sp_rank == 0:
        export_to_video(render_video, f"{output_path}/render.mp4", fps=16)
        export_to_video(mask_video, f"{output_path}/render_mask.mp4", fps=16)

    dist.barrier()

    render_frames = load_video(f"{output_path}/render.mp4")
    render_video = torch.stack([transforms.ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
    # replace the first frame with the high-quality reference image
    render_video[0, 0] = image
    render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
    render_mask_frames = load_video(f"{output_path}/render_mask.mp4")
    render_mask = torch.stack([transforms.ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
    render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]  # [0,1],[1,f,1,h,w]
    render_video = einops.rearrange(render_video, "b f c h w -> b c f h w").float()
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w").float()
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1

    camera_embedding = get_camera_embedding(intrinsics, w2cs, intrinsics.shape[0], height, width, normalize=True, is_w2c=True)

    # load text
    prompt = json.load(open(f"{input_path}/prompt.json"))["long caption"]

    meta_info = {
        "image": pil_image,
        "height": height,
        "width": width,
        "render_video": render_video.to(device),
        "render_mask": render_mask.to(device),
        "camera_embedding": camera_embedding.to(device),
        "prompt": prompt,
        "extrinsics": w2cs.to(device),
        "intrinsics": intrinsics.to(device)
    }

    if model_type in ("worldstereo-memory", "worldstereo-memory-dmd"):
        # Treat the first frame as the reference view
        extrinsic, extrinsic_ref = unified_camera_normalization(w2cs, w2cs[0:1])
        # Convert R matrix (w2c) to quaternion and concat with t vector
        R_matrix = extrinsic[:, :3, :3]
        t_vector = extrinsic[:, :3, 3]
        quaternion = matrix_to_quaternion(R_matrix)
        camera_qt = torch.cat([quaternion, t_vector], dim=-1).unsqueeze(0)  # [1, f, 7]

        R_matrix_ref = extrinsic_ref[:, :3, :3]
        t_vector_ref = extrinsic_ref[:, :3, 3]
        quaternion_ref = matrix_to_quaternion(R_matrix_ref)
        camera_qt_ref = torch.cat([quaternion_ref, t_vector_ref], dim=-1).unsqueeze(0)  # [1, f_ref, 7]

        meta_info["camera_qt"] = camera_qt.to(device)
        meta_info["camera_qt_ref"] = camera_qt_ref.to(device)

        reference_video = image[None].to(device)  # [1,c,h,w]
        meta_info["reference_video"] = einops.rearrange(reference_video[None], "b f c h w -> b c f h w")
        meta_info["ref_index"] = torch.tensor([0], device=device, dtype=torch.long)

    return meta_info
