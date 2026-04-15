import numpy as np
import torch
import torch.nn.functional as F
import random
import kornia
import imageio
import math
import os
import cv2
import inspect
import loguru
from PIL import Image, ImageDraw, ImageFont
import time
from collections import defaultdict
from contextlib import contextmanager
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import io
from decord import VideoReader, cpu


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def param_to_str(param_num):
    if param_num < 1e8:
        param_num = param_num / 1e6
        return f"{param_num:.3f}M"
    else:
        param_num = param_num / 1e9
        return f"{param_num:.3f}B"


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_boundaries_mask(disparity, sobel_threshold=0.3):
    def sobel_filter(disp, mode="sobel", beta=10.0):
        sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
        sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

        return alpha

    sobel_beta = 10.0
    normalized_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    return sobel_filter(normalized_disparity, "sobel", beta=sobel_beta) < sobel_threshold


def compute_normal_angles(normals1, normals2, eps=1e-8):
    """
    Compute the per-pixel angle (in degrees) between two normal maps.

    Args:
        normals1: first normal map, shape (H, W, 3) or (B, H, W, 3)
        normals2: second normal map, same shape as normals1
        eps: small value to prevent division by zero

    Returns:
        angles: angle matrix, shape (H, W) or (B, H, W), in degrees
    """
    # Verify input shapes match
    assert normals1.shape == normals2.shape

    # Compute dot product: (x1*x2 + y1*y2 + z1*z2)
    dot_product = torch.sum(normals1 * normals2, dim=-1)

    # Compute vector norms
    norm1 = torch.norm(normals1, dim=-1)
    norm2 = torch.norm(normals2, dim=-1)

    # Compute cosine (guard against division by zero)
    cos_theta = dot_product / (norm1 * norm2 + eps)

    # Clamp to [-1, 1] to handle numerical overflow
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle in radians and convert to degrees
    angles_rad = torch.acos(cos_theta)
    angles_deg = torch.rad2deg(angles_rad)

    return angles_deg


def erosion(image, kernel_size=3, padding=1):
    """
    Perform morphological erosion.

    Args:
        image: input image tensor, shape (batch, channels, height, width)
        kernel_size: structuring element size
        padding: padding size (default 1 keeps input/output dimensions equal)

    Returns:
        eroded image tensor
    """
    # Ensure input is a 4D tensor (batch, channels, height, width)
    if len(image.shape) != 4:
        raise ValueError("Input image must be a 4D tensor (batch, channels, height, width)")

    batch, channels, height, width = image.shape

    # Pad the image
    padded = F.pad(image, (padding, padding, padding, padding), mode='constant', value=1.0)

    # Extract sliding windows (unfold)
    # Result shape: (batch, channels, kernel_size*kernel_size, new_height, new_width)
    unfolded = F.unfold(padded, kernel_size=kernel_size, stride=1)
    unfolded = unfolded.view(batch, channels, kernel_size * kernel_size, height, width)

    # Take the minimum value in each window (core of erosion)
    eroded = torch.min(unfolded, dim=2)[0]

    return eroded


def point_padding(points):
    pad = torch.ones_like(points)[..., 0:1]
    return torch.cat([points, pad], dim=-1)


def np_point_padding(points):
    pad = np.ones_like(points)[..., 0:1]
    return np.concatenate([points, pad], axis=-1)


def add_text_to_corner(img, text, y=10, font_size=20, color="red"):
    """
    Add colored text to the top-left corner of an image.

    Args:
        img: PIL Image
        text: text string to draw
        y: vertical offset from top
        font_size: font size
        color: text color ("red" or "blue")
    """
    draw = ImageDraw.Draw(img)

    # Use default font (no path specified)
    font = ImageFont.load_default(size=font_size)

    # Position: top-left corner with a small margin (x=10, y=y)
    position = (10, y)

    # Text color (RGB)
    if color == "red":
        text_color = (255, 0, 0)
    elif color == "blue":
        text_color = (0, 0, 255)
    else:
        raise NotImplementedError

    # Draw text
    draw.text(position, text, font=font, fill=text_color)

    return img


def load_video(video_path):
    # Disable OpenCV internal multithreading to prevent CPU contention in multi-process environments
    cv2.setNumThreads(0)

    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            return []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)

    except Exception as e:
        print(f"Error reading video {video_path}: {e}")

    finally:
        # Always release the file handle regardless of errors
        cap.release()

    return frames


def get_last_video_frame(video_path):
    """Use decord's random access to read the last frame directly by index."""
    vr = VideoReader(video_path, ctx=cpu(0))
    last_frame = vr[-1].asnumpy()  # Index the last frame directly with an internal seek
    return last_frame


def save_video(
        frames: np.ndarray,
        output_path: str,
        fps: int = 24,
        codec: str = "libx264"  # universal MP4 codec with best compatibility
):
    """
    Save video frames to file at the specified FPS using imageio.

    Args:
        frames: input frame array, shape must be (f, h, w, c) where:
                f=frames, h=height, w=width, c=channels (1 or 3).
                dtype: float32 (values 0~1) or uint8 (values 0~255)
        output_path: output video path (e.g. ./output.mp4)
        fps: frames per second
        codec: video codec (libx264 default for MP4, best compatibility)
    """
    # 1. Validate input shape
    if len(frames.shape) != 4:
        raise ValueError(f"Input frame shape must be (f, h, w, c), got: {frames.shape}")
    f, h, w, c = frames.shape
    if c not in [1, 3]:
        raise ValueError(f"Channels must be 1 (grayscale) or 3 (color), got: {c}")

    # 2. Normalize to uint8 (0~255)
    processed_frames = frames.copy()
    if processed_frames.dtype == np.float32:
        # float32 (0~1) -> uint8 (0~255): clip first to avoid out-of-range values
        processed_frames = np.clip(processed_frames, 0.0, 1.0)
        processed_frames = (processed_frames * 255).astype(np.uint8)
    elif processed_frames.dtype == np.uint8:
        # uint8: just clip (0~255)
        processed_frames = np.clip(processed_frames, 0, 255)
    else:
        raise TypeError(f"Only float32/uint8 types supported, got: {processed_frames.dtype}")

    # 3. Expand single-channel (grayscale) to 3 channels (required by codec)
    if c == 1:
        processed_frames = np.repeat(processed_frames, 3, axis=-1)  # (f,h,w,1) -> (f,h,w,3)

    # 4. Write video at the specified FPS
    with imageio.get_writer(output_path, fps=fps, codec=codec) as writer:
        for idx, frame in enumerate(processed_frames):
            writer.append_data(frame)


def load_16bit_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def save_16bit_png_depth(depth: np.ndarray, depth_png: str):
    # Ensure the numpy array's dtype is float32, then cast to float16, and finally reinterpret as uint16
    depth_uint16 = np.array(depth, dtype=np.float32).astype(np.float16).view(np.uint16)

    # Create a PIL Image from the 16-bit depth values and save it
    depth_pil = Image.fromarray(depth_uint16)

    if not depth_png.endswith(".png"):
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError

    try:
        depth_pil.save(depth_png)
    except:
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError


def adjust_image_size(h, w):
    """
    Adjust h and w so that:
    1. h_new is divisible by 16
    2. w_new is divisible by 16
    3. (h_new//16) * (w_new//16) is divisible by 8

    Returns h_new and w_new as close to the original dimensions as possible (rounded up).
    """
    # Round h up to the nearest multiple of 16
    h_new = math.ceil(h / 16) * 16

    # Count the power of 2 in a = h_new // 16
    a = h_new // 16
    p = 0
    temp = a
    while temp > 0 and temp % 2 == 0:
        p += 1
        temp //= 2

    # b = w_new//16 must be divisible by 2^(3-p) to ensure a*b is divisible by 8
    required_factor = 1 << max(0, 3 - p)  # 2^max(0, 3-p)

    # Round w_new up to the smallest value satisfying the constraint
    b = math.ceil(w / 16)
    b = math.ceil(b / required_factor) * required_factor
    w_new = b * 16

    return h_new, w_new


def rank0_log(message, level="INFO"):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.opt(depth=1).log(level, message)


class Timer:
    """Concise multi-section timer."""

    def __init__(self):
        self.records = defaultdict(list)
        self._start_times = {}

    def start(self, name: str):
        """Start timing a named section."""
        self._start_times[name] = time.perf_counter()

    def end(self, name: str):
        """Stop timing a named section."""
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.records[name].append(elapsed)
            del self._start_times[name]
            return elapsed
        return 0

    @contextmanager
    def track(self, name: str):
        """Context manager for timing a code block."""
        self.start(name)
        try:
            yield
        finally:
            self.end(name)

    def summary(self):
        """Print a statistics summary, separately for IO and non-IO sections."""

        # Categorize sections
        io_total_time = 0.0
        non_io_total_time = 0.0
        io_records = {}
        non_io_records = {}

        for name, times in self.records.items():
            total_time = sum(times)
            if "[IO]" in name:
                io_total_time += total_time
                io_records[name] = times
            else:
                non_io_total_time += total_time
                non_io_records[name] = times

        overall_total = io_total_time + non_io_total_time

        # Print header
        print("\n" + "=" * 80)
        print(f"{'Part':<35} {'Calls':>8} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("=" * 80)

        # Print non-IO sections
        if non_io_records:
            print("-" * 80)
            print("[Compute Parts]")
            print("-" * 80)
            for name, times in non_io_records.items():
                print(
                    f"{name:<35} {len(times):>8} {sum(times):>10.4f} "
                    f"{sum(times) / len(times):>10.4f} {min(times):>10.4f} {max(times):>10.4f}"
                )

        # Print IO sections
        if io_records:
            print("-" * 80)
            print("[IO Parts]")
            print("-" * 80)
            for name, times in io_records.items():
                print(
                    f"{name:<35} {len(times):>8} {sum(times):>10.4f} "
                    f"{sum(times) / len(times):>10.4f} {min(times):>10.4f} {max(times):>10.4f}"
                )

        # Print summary
        print("=" * 80)
        print(f"{'[Summary] Compute Total':<35} {non_io_total_time:>10.4f}s ({non_io_total_time / overall_total * 100 if overall_total > 0 else 0:>6.2f}%)")
        print(f"{'[Summary] IO Total':<35} {io_total_time:>10.4f}s ({io_total_time / overall_total * 100 if overall_total > 0 else 0:>6.2f}%)")
        print(f"{'[Summary] Overall Total':<35} {overall_total:>10.4f}s")
        print("=" * 80)



def split_n_into_d_parts(N: int, D: int) -> list[int]:
    """
    Split integer N evenly into D parts, minimizing the difference between parts (at most 1).

    Args:
        N: non-negative integer to split (total count)
        D: number of parts (positive integer)

    Returns:
        list of integers of length D that sum to N

    Raises:
        ValueError: if D is not a positive integer or N is not an integer
    """
    # Validate: D must be a positive integer
    if not isinstance(D, int) or D <= 0:
        raise ValueError
    # Handle non-integer N
    if not isinstance(N, int):
        raise ValueError

    # Core logic: base value + remainder
    q, r = divmod(N, D)  # equivalent to q=N//D, r=N%D
    # First (D-r) parts are q, last r parts are q+1, ensuring sum=N with minimal difference
    result = [q] * (D - r) + [q + 1] * r
    return result

def color_print(msg, level="info"):
    """
    Minimal colored print utility.

    Args:
        msg: message string to print
        level: print level — info (green) / error (red) / warning (yellow), default info
    """
    # Color code map; add entries here to extend levels
    COLOR_MAP = {
        "info": "\033[32m",     # green: normal
        "error": "\033[31m",    # red: failure / critical
        "warning": "\033[33m"   # yellow: warning (optional extension)
    }
    COLOR_RESET = "\033[0m"  # reset color
    print(f"{COLOR_MAP.get(level, COLOR_MAP['info'])}{msg}{COLOR_RESET}")


def flatten(data):
    result = []
    # If the outermost element is None, return [None] by convention
    if data is None:
        return [None]

    for item in data:
        # Only recurse if the element is a list
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            # item may be a value like 'valid' or None
            result.append(item)
    return result

def sample_align_nframe(N, n):
    if n > (N - 1):
        raise ValueError

    # Core logic: there are N-1 available positions (1 to N-1), must include N-1,
    # with spacing decreasing from front to back.
    # We need to select n points with n-1 internal gaps and 1 initial offset.
    # Total free units = (N-1) - n
    extra_space = (N - 1) - n

    # Distribute free units to the start position and front-loaded gaps.
    # To keep larger gaps at the front, allocate extra_space preferentially to
    # the first positions.

    # Allocatable slots: 1 start + (n-1) gaps = n slots total
    num_allocatable_slots = n
    base_extra = extra_space // num_allocatable_slots
    remainder = extra_space % num_allocatable_slots

    # Build allocation sequence (larger increments first)
    allocations = [base_extra + 1] * remainder + [base_extra] * (num_allocatable_slots - remainder)

    # Generate indices
    indices = []
    # First point: starts at 1 + first allocation
    current = 1 + allocations[0]
    indices.append(current)

    # Subsequent points: step = 1 (base step) + allocation
    for i in range(1, len(allocations)):
        current += (1 + allocations[i])
        indices.append(current)

    return np.array(indices)


def colorize_depth(
        depth,
        colormap='plasma',
        min_depth=None,
        max_depth=None,
        inverse=True,
        save_path=None,
        return_pil=False,
        show_colorbar=False,
        colorbar_label='Depth',
        colorbar_width=0.03,  # colorbar width ratio
        colorbar_pad=0.02,    # colorbar padding from image
        colorbar_ticks=5,     # number of colorbar ticks
        figsize_scale=1.0,    # image scaling factor
        dpi=150,              # DPI when saving
        title=None,           # optional title
):
    """
    Colorize a depth map for visualization.

    Args:
        depth: [H,W] or [1,H,W] or [B,1,H,W], numpy array or torch tensor
        colormap: 'plasma', 'turbo', 'inferno', 'magma', 'viridis', 'jet'
        min_depth: minimum depth value for normalization (auto if None)
        max_depth: maximum depth value for normalization (auto if None)
        inverse: True = near regions red, far regions purple
        save_path: path to save the output image
        return_pil: True returns PIL Image, False returns numpy array
        show_colorbar: whether to display a colorbar
        colorbar_label: colorbar label text
        colorbar_width: colorbar width as fraction of image
        colorbar_pad: padding between colorbar and image
        colorbar_ticks: number of ticks on colorbar
        figsize_scale: image scaling factor
        dpi: dots per inch when saving
        title: optional title string

    Returns:
        colorized depth map (numpy array or PIL Image)
    """
    # Convert to numpy [H, W]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = np.squeeze(depth)

    # Ensure 2D
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got shape {depth.shape}")

    # Get valid depth range
    valid = (depth > 0) & np.isfinite(depth)

    if min_depth is None:
        min_depth = depth[valid].min().item()
    if max_depth is None:
        max_depth = depth[valid].max().item()

    # Normalize
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Invert color direction
    if inverse:
        depth_norm = 1.0 - depth_norm

    # Zero out invalid regions
    depth_norm[~valid] = 0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_norm)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    # ===== Without colorbar (original logic) =====
    if not show_colorbar:
        if save_path:
            img = Image.fromarray(colored)
            img.save(save_path)

        if return_pil:
            return Image.fromarray(colored)
        return colored

    # ===== With colorbar =====
    h, w = depth.shape
    figsize = (w / 100 * figsize_scale, h / 100 * figsize_scale)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display depth map (set invalid region to NaN so it renders as blank)
    depth_display = depth.copy()
    depth_display[~valid] = np.nan

    # Create Normalize object
    if inverse:
        # When inverted, also invert the colorbar mapping
        norm = Normalize(vmin=max_depth, vmax=min_depth)
    else:
        norm = Normalize(vmin=min_depth, vmax=max_depth)

    im = ax.imshow(depth_display, cmap=colormap, norm=norm)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=12)

    # Add colorbar
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=colorbar_width,
        pad=colorbar_pad,
        label=colorbar_label
    )

    # Set colorbar ticks
    tick_values = np.linspace(min_depth, max_depth, colorbar_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{v:.2f}' for v in tick_values])

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

    # Convert to numpy array or PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    result_img = Image.open(buf)

    plt.close(fig)

    if return_pil:
        return result_img
    return np.array(result_img)[:, :, :3]