import argparse
import os
from glob import glob

import imagesize
from tqdm import tqdm
import json
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from src.general_utils import set_seed, rank0_log, load_video
from src.sp_utils.parallel_states import initialize_parallel_state
from models.worldstereo_wrapper import WorldStereo
from src.data_utils import sort_trajs, recon_sort_trajs, load_mutli_traj_dataset
from diffusers.utils import export_to_video
from src.retrieval_wm import SimpleMemoryBank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="worldstereo-memory-dmd", choices=["worldstereo-memory", "worldstereo-memory-dmd"],
                        help="Model type (e.g., 'worldstereo-memory', 'worldstereo-memory-dmd')")
    parser.add_argument("--task_type", type=str, default="panorama", choices=["panorama", "reconstruction"], help="task type")
    parser.add_argument("--input_path", type=str, default="examples/panorama", help="panoramic input path")
    parser.add_argument("--output_path", type=str, default="outputs", help="Target path for output results")
    parser.add_argument("--align_nframe", type=int, default=8, help="Saving number of frames for each video clip")
    parser.add_argument("--local_files_only", action="store_true", help="If True, avoid downloading the file and return the path to the local cached file if it exists.")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP model sharding")
    parser.add_argument("--seed", default=1024, type=int, help="Random seed")
    args = parser.parse_args()

    # ── distributed setup ─────────────────────────────────────────────
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
    )
    device_num = torch.cuda.device_count()
    device_mesh = init_device_mesh(
        "cuda",
        (world_size // device_num, device_num),
        mesh_dim_names=("rep", "shard"),
    )
    rank0_log(f"World size: {world_size}")

    # ── sequence parallel ──────────────────────────────────────────────
    parallel_dims = initialize_parallel_state(sp=world_size)
    sp_size = parallel_dims.sp if parallel_dims.sp_enabled else 1
    sp_rank = parallel_dims.sp_rank if parallel_dims.sp_enabled else 0
    data_rank = dist.get_rank() // sp_size

    global_seed = args.seed + data_rank
    set_seed(global_seed)
    print(f"Global rank:{dist.get_rank()}, Local rank:{local_rank}, "
          f"SP_rank:{sp_rank}, SP_group:{data_rank}, seed:{global_seed}.")

    # ── model loading ──────────────────────────────────────────────────
    torch.set_default_dtype(torch.float32)

    worldstereo = WorldStereo.from_pretrained(
        "hanshanxue/WorldStereo",
        subfolder=args.model_type,
        local_files_only=args.local_files_only,
        sp_world_size=sp_size,
        fsdp=args.fsdp,
        device_mesh=device_mesh,
        device=device,
    )

    # ── inference ──────────────────────────────────────────────────────
    dist.barrier()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Auto-select autocast precision: prefer bf16, then fp16, fall back to fp32 (disable autocast)
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability(device)[0] >= 7:  # fp16 requires SM >= 70
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None  # no half-precision support, fall back to fp32
    rank0_log(f"Autocast dtype: {autocast_dtype if autocast_dtype else 'disabled (fp32)'}")

    # load data
    if os.path.exists(f"{args.input_path}/pano_bank") or os.path.exists(f"{args.input_path}/image.png"):
        scene_list = [args.input_path]
    else:
        scene_list = glob(f"{args.input_path}/*")
    scene_list.sort()
    rank0_log(f"Building dataset. {len(scene_list)} scenes found.")

    for scene_path in tqdm(scene_list):
        scene_name = os.path.basename(scene_path)
        rank0_log(f"Processing scene {scene_name}.")
        output_path = f"{args.output_path}/{scene_name}"
        if rank == 0:
            os.makedirs(output_path, exist_ok=True)

        if args.task_type == "panorama":
            width, height = imagesize.get(f"{scene_path}/pano_bank/images/0000.png")
        else:
            width, height = imagesize.get(f"{scene_path}/image.png")
        with torch.no_grad():
            memory_bank = SimpleMemoryBank(cfg=worldstereo.cfg, root_path=scene_path, image_width=width, image_height=height, device=device,
                                           max_reference=8, align_nframe=args.align_nframe, rank=sp_rank, world_size=sp_size)

            if args.task_type == "panorama":
                render_list = sort_trajs(scene_path)
            else:
                render_list = recon_sort_trajs(scene_path)
            rank0_log(f"Scene {scene_name}: {len(render_list)} renderings found.")

        for render_path in render_list:
            view_id, traj_id = render_path.split('/')[-3], render_path.split('/')[-2]
            if args.task_type == "reconstruction":
                view_id = "renders"
            rank0_log(f"Scene {scene_name}: view: {view_id}, traj: {traj_id}.")

            target_cameras = json.load(open(f"{scene_path}/{view_id}/{traj_id}/camera.json"))
            tar_w2cs = torch.from_numpy(np.array(target_cameras["extrinsic"])).to(dtype=torch.float32, device=device)
            tar_Ks = torch.from_numpy(np.array(target_cameras["intrinsic"])).to(dtype=torch.float32, device=device)

            if os.path.exists(f"{output_path}/{view_id}/{traj_id}/{args.model_type}_result.mp4"):
                # Only update the memory bank, skip inference
                gen_frames = load_video(f"{output_path}/{view_id}/{traj_id}/{args.model_type}_result.mp4")
                memory_bank.update_memory(gen_frames=gen_frames, tar_w2cs_full=tar_w2cs, tar_Ks_full=tar_Ks, view_id=view_id, traj_id=traj_id)
                rank0_log(f"View: {view_id}, traj: {traj_id} is already exist. Updating memory bank and skipping.")
                continue

            # retrieval and save related data
            retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = memory_bank.retrieval(tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id)
            if rank == 0:  # rank 0 saves retrieval results
                os.makedirs(f"{output_path}/{view_id}/{traj_id}/memory_inputs", exist_ok=True)
                export_to_video(retrieved_frames / 255, f"{output_path}/{view_id}/{traj_id}/memory_inputs/{args.model_type}.mp4", fps=16)
                if ref_index_dict is not None:
                    with open(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{args.model_type}_ref_index.json", "w") as w:
                        json.dump(ref_index_dict, w, indent=2)
                if ref_w2cs is not None:
                    ref_w2cs = ref_w2cs.cpu().numpy().tolist()
                    with open(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{args.model_type}_ref_w2cs.json", "w") as w:
                        json.dump(ref_w2cs, w, indent=2)

            dist.barrier()

            # prepare inputs
            meta_data = load_mutli_traj_dataset(cfg=worldstereo.cfg, input_path=scene_path, output_path=output_path, view_id=view_id, traj_id=traj_id,
                                                device=device, ref_index=ref_index, model_type=args.model_type, task_type=args.task_type)

            pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
            pipeline_kwargs.update(
                negative_prompt=worldstereo.cfg.get("negative_prompt", ""),
                generator=generator,
                output_type="pt",
                latent_cond_mode=worldstereo.cfg.latent_cond_mode,
            )

            if args.model_type == "worldstereo-memory-dmd":
                pipeline_kwargs["mode"] = "test"
            else:
                pipeline_kwargs["guidance_scale"] = 5.0

            # pipeline inference
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                output = worldstereo.pipeline(**pipeline_kwargs).frames[0].float()
            output = output.cpu().permute(0, 2, 3, 1).numpy()
            torch.cuda.empty_cache()

            if rank == 0:
                export_to_video(output, f"{output_path}/{view_id}/{traj_id}/{args.model_type}_result.mp4", fps=16)

            dist.barrier()  # wait for save to complete
            # update memory bank
            gen_frames = load_video(f"{output_path}/{view_id}/{traj_id}/{args.model_type}_result.mp4")
            memory_bank.update_memory(gen_frames=gen_frames, tar_w2cs_full=tar_w2cs, tar_Ks_full=tar_Ks, view_id=view_id, traj_id=traj_id)
            dist.barrier()  # wait for update to complete

        # Convert to WorldMirror input format
        memory_bank.apply_worldmirror(f"{output_path}/world_mirror_data/{args.model_type}", skip_exist=True)

    if dist.is_initialized():
        dist.destroy_process_group()
