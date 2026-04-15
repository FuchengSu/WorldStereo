<div align="center">

<h2 align="center">
  WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories
</h2>

<p align="center">
  <strong>Yisu Zhang</strong><sup>1,2*</sup> &nbsp;
  <strong>Chenjie Cao</strong><sup>2*</sup> &nbsp;
  <strong>Tengfei Wang</strong><sup>2†</sup> &nbsp;
  <strong>Xuhui Zuo</strong><sup>2</sup> &nbsp;
  <strong>Junta Wu</strong><sup>2</sup> &nbsp;
  <strong>Jianke Zhu</strong><sup>1‡</sup> &nbsp;
  <strong>Chunchao Guo</strong><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>Zhejiang University &nbsp;&nbsp;
  <sup>2</sup>Tencent Hunyuan
  <br>
  <small>*Equal Contribution &nbsp; †Project Lead &nbsp; ‡Corresponding Author</small>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.24233"><img src="https://img.shields.io/badge/arXiv-2602.24233-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/hanshanxue/WorldStereo"><img src="https://img.shields.io/badge/🤗-Model_Weights-yellow.svg" alt="HuggingFace"></a>
</p>

</div>

---

## 📅 News

- `[2026.03]` 📄 Paper is now available on arXiv: [https://arxiv.org/abs/2602.24233](https://arxiv.org/abs/2602.24233)
- `[2026.04]` 🚀 Code and model weights of WorldStereo 2.0 are now released!

---

## ✅ TODO List

- [x] Release inference code and model weights of WorldStereo 2.0
- [ ] Release data pre-processing pipelines for panoramic and multi-trajectory scenes

---

## 📖 Abstract

We propose **WorldStereo**, a novel framework that bridges camera-guided video generation and 3D reconstruction via two dedicated geometric memory modules:

- **Global-Geometric Memory (GGM)** enables precise camera control while injecting coarse structural priors through incrementally updated point clouds via a ControlNet branch.
- **Spatial-Stereo Memory (SSM)** constrains the model's attention receptive fields with 3D correspondences to focus on fine-grained details from the memory bank.

Together, these components enable WorldStereo to generate multi-view-consistent videos under precise camera control, facilitating high-quality 3D reconstruction. Furthermore, WorldStereo shows impressive efficiency by leveraging a distribution-matching distilled (DMD) VDM backbone without joint training.

---

## 🎬 Results

### 3D Reconstruction from a Single Image

Given a single reference image, WorldStereo generates multi-view consistent videos and reconstructs a dense 3D point cloud. Below are example results on two scenes.

**Scene: Kitchen** &nbsp; | &nbsp; Input image → Point cloud (5 views)

<p align="center">
  <img src="examples/reconstruction/kitchen/image.png" height="120" alt="Kitchen input">
  &nbsp;
  <img src="assets/kitchen-1.jpg" height="120" alt="Kitchen pcd 1">
  <img src="assets/kitchen-2.jpg" height="120" alt="Kitchen pcd 2">
  <img src="assets/kitchen-3.jpg" height="120" alt="Kitchen pcd 3">
  <img src="assets/kitchen-4.jpg" height="120" alt="Kitchen pcd 4">
  <img src="assets/kitchen-5.jpg" height="120" alt="Kitchen pcd 5">
</p>

### Camera Control

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3" align="center">Camera Metrics</th>
      <th colspan="4" align="center">Visual Quality</th>
    </tr>
    <tr>
      <th>RotErr ↓</th><th>TransErr ↓</th><th>ATE ↓</th>
      <th>Q-Align ↑</th><th>CLIP-IQA+ ↑</th><th>Laion-Aes ↑</th><th>CLIP-I ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>SEVA</td><td>1.690</td><td>1.578</td><td>2.879</td><td>3.232</td><td>0.479</td><td>4.623</td><td>77.16</td></tr>
    <tr><td>Gen3C</td><td>0.944</td><td>1.580</td><td>2.789</td><td>3.353</td><td>0.489</td><td>4.863</td><td>82.33</td></tr>
    <tr><td>WorldStereo</td><td>0.762</td><td>1.245</td><td>2.141</td><td>4.149</td><td><b>0.547</b></td><td>5.257</td><td>89.05</td></tr>
    <tr><td><b>WorldStereo 2.0</b></td><td><b>0.492</b></td><td><b>0.968</b></td><td><b>1.768</b></td><td><b>4.205</b></td><td>0.544</td><td><b>5.266</b></td><td><b>89.43</b></td></tr>
  </tbody>
</table>

### Single-View-Generated Reconstruction

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="4">Tanks-and-Temples</th>
      <th colspan="4">MipNeRF360</th>
    </tr>
    <tr>
      <th>Precision ↑</th>
      <th>Recall ↑</th>
      <th>F1-Score ↑</th>
      <th>AUC ↑</th>
      <th>Precision ↑</th>
      <th>Recall ↑</th>
      <th>F1-Score ↑</th>
      <th>AUC ↑</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">SEVA</td>
      <td>33.59</td>
      <td>35.34</td>
      <td>36.73</td>
      <td>51.03</td>
      <td>22.38</td>
      <td>55.63</td>
      <td>28.75</td>
      <td>46.81</td>
    </tr>
    <tr>
      <td align="left">Gen3C</td>
      <td><u>46.73</u></td>
      <td>25.51</td>
      <td>31.24</td>
      <td>42.44</td>
      <td>23.28</td>
      <td><strong>75.37</strong></td>
      <td>35.26</td>
      <td>52.10</td>
    </tr>
    <tr>
      <td align="left">Lyra</td>
      <td><strong>50.38</strong></td>
      <td>28.67</td>
      <td>32.54</td>
      <td>43.05</td>
      <td>30.02</td>
      <td>58.60</td>
      <td>36.05</td>
      <td>49.89</td>
    </tr>
    <tr>
      <td align="left">FlashWorld</td>
      <td>26.58</td>
      <td>20.72</td>
      <td>22.29</td>
      <td>30.45</td>
      <td>35.97</td>
      <td>53.77</td>
      <td>42.60</td>
      <td>53.86</td>
    </tr>
    <tr>
      <td align="left">WorldStereo 2.0</td>
      <td>43.62</td>
      <td><u>41.02</u></td>
      <td><u>41.43</u></td>
      <td><u>58.19</u></td>
      <td><strong>43.19</strong></td>
      <td><u>65.32</u></td>
      <td><strong>51.27</strong></td>
      <td><strong>65.79</strong></td>
    </tr>
    <tr>
      <td align="left">WorldStereo 2.0 (DMD)</td>
      <td>40.41</td>
      <td><strong>44.41</strong></td>
      <td><strong>43.16</strong></td>
      <td><strong>60.09</strong></td>
      <td><u>42.34</u></td>
      <td>64.83</td>
      <td><u>50.52</u></td>
      <td><u>65.64</u></td>
    </tr>
  </tbody>
</table>

---

## 🆕 WorldStereo 2.0 vs. 1.0

WorldStereo 2.0 introduces four key improvements over the original version:

| | WorldStereo 1.0 | WorldStereo 2.0                                                                                                                                                                                           |
|---|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Latent Space** | Standard video latent space | **Keyframe latent space** — encodes each frame independently, substantially improving visual quality of generated novel views and completely supporting parallel encoding/decoding                        |
| **Memory Mechanism** | Cross-attention to retrieved reference frames | **Stereo stitching in the main branch** — reference views are spatially concatenated with target frames along the width dimension in the main DiT branch, enabling stronger and more direct memory fusion |
| **Backbone Fine-tuning** | Frozen backbone | **Partial backbone fine-tuning** — backbone weights are selectively updated to adapt to the keyframe latent space and improve overall generation quality                                                  |
| **Training Data** | Limited camera trajectories | **Expanded UE rendering data** — significantly more Unreal Engine rendered scenes with diverse and precise camera motions, leading to stronger camera control and memory capabilities                     |

 More details of WorldStereo 2.0 are shown in [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0).

---

## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/FuchengSu/WorldStereo.git
cd WorldStereo
```

**2. Install core dependencies:**
```bash
conda create -n worldstereo python=3.11
conda activate worldstereo
pip install -r requirements.txt
```

**3. Install PyTorch3D (required for point cloud rendering):**
```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

**4. Install MoGe (monocular depth estimation):**
```bash
pip install git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6
```

**5. (Optional) Install third-party reconstruction module for WorldMirror reconstruction:**
```bash
mkdir third_party
cd third_party
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git
pip install -r HY-World-2.0/requirements.txt
```

> **Note:** `third_party/HY-World-2.0` is required only for `apply_worldmirror` post-processing (multi-view depth consistency and Gaussian Splatting reconstruction). You can skip it for basic video generation.

---

## 🚀 Quick Start

### Model Variants

WorldStereo ships three model variants, each suited to a different use case:

| Model Type | Entry Point | Description |
|---|---|---|
| `worldstereo-camera` | `run_camera_control.py` | Camera control only; single-view input |
| `worldstereo-memory` | `run_camera_control.py` / `run_multi_traj.py` | Full model with GGM + SSM; multi-view consistent generation; best quality |
| `worldstereo-memory-dmd` | `run_camera_control.py` / `run_multi_traj.py` | DMD distillation variant; 4-step inference, fastest |

Models are automatically downloaded from HuggingFace Hub ([`hanshanxue/WorldStereo`](https://huggingface.co/hanshanxue/WorldStereo)) on first run.

---

### Single-View Camera Control

Generate a video from a single image along a specified camera trajectory:

```bash
python run_camera_control.py \
    --model_type worldstereo-camera \
    --input_path examples/images \
    --output_path outputs \
    --seed 1024
```

---

### Multi-GPU Inference (Sequence Parallel)

Scale to multiple GPUs using Sequence Parallelism (SP) and FSDP:

```bash
torchrun --nproc_per_node=8 run_camera_control.py \
    --model_type worldstereo-memory \
    --input_path examples/panorama \
    --output_path outputs \
    --fsdp
```

---

### Multi-Trajectory Inference (Panorama / Reconstruction)

For panoramic scene generation or 3D reconstruction from multiple trajectories:

```bash
# Panoramic scene generation
torchrun --nproc_per_node=8 run_multi_traj.py \
    --model_type worldstereo-memory \
    --task_type panorama \
    --input_path examples/panorama \
    --output_path outputs \
    --fsdp

# Panoramic scene generation (DMD fast variant)
torchrun --nproc_per_node=8 run_multi_traj.py \
    --model_type worldstereo-memory-dmd \
    --task_type panorama \
    --input_path examples/panorama \
    --output_path outputs \
    --fsdp

# 3D scene reconstruction
torchrun --nproc_per_node=8 run_multi_traj.py \
    --model_type worldstereo-memory \
    --task_type reconstruction \
    --input_path examples/reconstruction \
    --output_path outputs \
    --fsdp

# 3D scene reconstruction (DMD fast variant)
torchrun --nproc_per_node=8 run_multi_traj.py \
    --model_type worldstereo-memory-dmd \
    --task_type reconstruction \
    --input_path examples/reconstruction \
    --output_path outputs \
    --fsdp
```

---

### WorldMirror 3D Reconstruction (Optional)

After running `run_multi_traj.py`, the memory bank is automatically exported to a WorldMirror-compatible format under `<output_path>/<scene>/world_mirror_data/<model_type>/`. You can then run feedforward 3D reconstruction with [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0):

```bash
# Requires: pip install -r third_party/HY-World-2.0/requirements.txt

cd third_party/HY-World-2.0
torchrun --nproc_per_node=8 -m hyworld2.worldrecon.pipeline --input_path ../../outputs/<scene>/world_mirror_data/<model_type>/images \
          --prior_cam_path ../../outputs/<scene>/world_mirror_data/<model_type>/cameras.json \
          --strict_output_path ../../outputs/<scene>/world_mirror_data/<model_type>/results \
          --target_size 832 --use_fsdp --enable_bf16 --no_save_normal --no_save_gs --no_sky_mask \
          --apply_edge_mask --apply_confidence_mask --confidence_percentile 15.0 --compress_pts --no_interactive \
          --disable_heads gs points
```

This produces metric-scale depth, surface normals, camera poses, a dense point cloud (`.ply`), and optionally Gaussian Splat renderings from the generated multi-view frames.

---

### Python API

```python
import torch
from models.worldstereo_wrapper import WorldStereo

device = torch.device("cuda:0")

worldstereo = WorldStereo.from_pretrained(
    "hanshanxue/WorldStereo",
    subfolder="worldstereo-memory",   # or "worldstereo-camera" / "worldstereo-memory-dmd"
    sp_world_size=1,
    fsdp=False,
    device=device,
)

output = worldstereo(**pipeline_inputs)
```

---

### CLI Reference

**`run_camera_control.py`**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_type` | `worldstereo-camera` | Model variant to use |
| `--input_path` | `examples/images` | Input scene directory |
| `--output_path` | `outputs` | Output directory |
| `--local_files_only` | `False` | Use locally cached weights instead of downloading |
| `--fsdp` | `False` | Enable FSDP model sharding |
| `--seed` | `1024` | Random seed |

**`run_multi_traj.py`** (additional flags)

| Flag | Default | Description |
|------|---------|-------------|
| `--task_type` | `panorama` | `panorama` or `reconstruction` |
| `--align_nframe` | `8` | Frames per clip saved for updating the memory bank |

---

## 📂 Input Data Format

### Camera-Only Inference (`examples/images/`)

```
<scene>/
├── image.png                 # reference image
├── prompt.json               # text descriptions at three verbosity levels
│   # {"short caption": ..., "medium caption": ..., "long caption": ...}
└── camera.json               # camera trajectory
    # {"motion_list": [...], "extrinsic": [...], "intrinsic": [...]}
```

### Memory-Augmented Multi-Trajectory (`examples/panorama/`, `examples/reconstruction/`)

```
<scene>/
├── panorama.png              # (optional) full panorama — triggers VLM single-path inference
├── meta_info.json            # {"scene_type": "perspective" | "panorama"}
├── start_frame.png           # reference start image for depth initialization
└── render_results/
    └── <view_id>/
        └── <traj_id>/
            ├── render.mp4         # pre-rendered geometry video (point cloud warp)
            ├── render_mask.mp4    # binary occlusion mask video
            └── camera.json        # {"extrinsic": [...], "intrinsic": [...]}
```

---

## 🔧 Architecture

### Model Variants

WorldStereo defines two transformer architectures in `models/worldstereo.py`, both extending `WanTransformer3DModel` from diffusers:

- **`WorldStereoModel`** — Wan DiT backbone + ControlNet. Used by `worldstereo-camera`. The ControlNet encodes rendered point cloud geometry and camera embeddings, injecting residuals at each transformer block.
- **`WorldStereoRefSModel`** — Extends `WorldStereoModel` with `WanTransformerSparseSpatialBlock` layers. These SSM blocks perform sparse attention over retrieved reference frames, guided by 3D correspondences. Used by `worldstereo-memory` and `worldstereo-memory-dmd`.

### Inference Pipelines

Three pipelines are provided under `models/pipelines/`, selected automatically based on `model_type` in the config:

| Pipeline | Class | Mode |
|----------|-------|------|
| `pipeline_pcd_keyframe.py` | `KFPCDControllerPipeline` | Camera; standard DDIM sampling |
| `pipeline_ref_keyframe.py` | `KFPCDControllerRefPipeline` | Camera + GGM + SSM; standard DDIM sampling |
| `pipeline_dmd_keyframe.py` | `RefKFDMDGeneratorPipeline` | Camera + GGM + SSM; 4-step DMD distillation |

### 3D Memory Bank

The memory bank (`src/retrieval_wm.py`) manages the growing 3D representation across trajectories:

1. **Init** — MoGe depth estimation on the start frame lifts it to a point cloud.
2. **Retrieve** — For each new target trajectory, the most relevant reference frames are selected via FOV-overlap scoring combined with DINOv2 image features and quality-aware furthest-point sampling.
3. **Update** — After generation, new frames and their estimated depths are appended to the bank.
4. **Reconstruction** — Feedforward reconstruction via HY-World 2.0 WorldMirror enforces multi-view depth consistency; final global alignment produces a unified point cloud.

### Distributed Inference

WorldStereo supports two parallelism strategies:

- **Sequence Parallel (SP)** — The sequence dimension is sharded across the SP group at each attention layer (`models/attention.py`). Controlled by `torchrun --nproc_per_node`.
- **FSDP** — Full-Sharded Data Parallel wraps both the transformer and auxiliary encoders. Enabled with `--fsdp`. Requires a `device_mesh` with `("rep", "shard")` dimensions.

---

## 🤝 Acknowledgements

WorldStereo builds upon the following excellent works:

- [Wan](https://github.com/wan-ai-tech/Wan2.1) — Video DiT backbone
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo) — Components of sequence parallel and video generation model
- [MoGe](https://github.com/microsoft/MoGe) — Monocular geometry estimation
- [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) — WorldMirror reconstruction module
- [diffusers](https://github.com/huggingface/diffusers) — Pipeline and model utilities

---

## 📝 Citation

If you find WorldStereo useful in your research, please cite:

```bibtex
@article{zhang2026worldstereo,
  title     = {WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories},
  author    = {Zhang, Yisu and Cao, Chenjie and Wang, Tengfei and Zuo, Xuhui and Wu, Junta and Zhu, Jianke and Guo, Chunchao},
  journal   = {arXiv preprint arXiv:2602.24233},
  year      = {2026}
}
```
