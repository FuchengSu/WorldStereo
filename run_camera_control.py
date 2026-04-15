import argparse
import os
from glob import glob
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from src.general_utils import set_seed, rank0_log
from src.sp_utils.parallel_states import initialize_parallel_state
from models.worldstereo_wrapper import WorldStereo
from moge.model.v2 import MoGeModel
from src.data_utils import load_single_view_data
from diffusers.utils import export_to_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="worldstereo-camera",
                        choices=["worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd"],
                        help="Model type (e.g., 'worldstereo-camera', 'worldstereo-memory', 'worldstereo-memory-dmd')")
    parser.add_argument("--input_path", type=str, default="examples/images", help="image input path")
    parser.add_argument("--output_path", type=str, default="outputs", help="Target path for output results")
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
    # moge is used for warp rendering
    depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

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
    if os.path.exists(f"{args.input_path}/image.png"):
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
        with torch.no_grad():
            meta_data = load_single_view_data(cfg=worldstereo.cfg, input_path=scene_path, output_path=output_path,
                                              model_type=args.model_type, depth_model=depth_model, device=device,
                                              sp_size=sp_size, sp_rank=sp_rank)

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

            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                output = worldstereo.pipeline(**pipeline_kwargs).frames[0].float()
            output = output.cpu().permute(0, 2, 3, 1).numpy()
            torch.cuda.empty_cache()

        if rank == 0:
            export_to_video(output, f"{output_path}/{args.model_type}_result.mp4", fps=16)


    if dist.is_initialized():
        dist.destroy_process_group()
