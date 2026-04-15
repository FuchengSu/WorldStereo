import collections
import torch
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import torch.distributed as dist
from src.general_utils import (
    rank0_log,
    color_print,
    sample_align_nframe,
)
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional, Literal
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from src.data_utils import assign_scale


def statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    KNN-based Statistical Outlier Removal.
    For each point, computes the mean distance to its K nearest neighbors.
    Points whose mean distance exceeds (global_mean + std_ratio * global_std) are treated as outliers.

    Args:
        points: [N, 3] np.ndarray, point cloud coordinates
        colors: [N, 3] np.ndarray, point cloud colors
        nb_neighbors: int, number of KNN neighbors
        std_ratio: float, standard deviation multiplier threshold

    Returns:
        filtered_points: [M, 3] np.ndarray, filtered point cloud coordinates
        filtered_colors: [M, 3] np.ndarray, filtered point cloud colors
        inlier_mask: [N] bool np.ndarray, inlier mask
    """
    if points.shape[0] <= nb_neighbors:
        return points, colors, np.ones(points.shape[0], dtype=bool)

    tree = cKDTree(points)
    # Query nb_neighbors+1 neighbors (including self), then use the distances of the last nb_neighbors
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    # dists[:, 0] is the self-distance (= 0); use dists[:, 1:] to compute mean distance
    mean_dists = np.mean(dists[:, 1:], axis=1)  # [N]

    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_ratio * global_std

    inlier_mask = mean_dists <= threshold
    return points[inlier_mask], colors[inlier_mask], inlier_mask


def compute_depth_percentile_map(depth, depth_mask):
    """
    Compute the depth percentile of each pixel within the valid region.

    Args:
        depth: [h, w] numpy array, depth map (near=small, far=large), valid values > 0
        depth_mask: [h, w] bool numpy array, True indicates valid region

    Returns:
        percentile_map: [h, w] numpy array, depth percentile (0-100) for each pixel;
                        pixels where mask is False are set to 0
    """
    h, w = depth.shape
    percentile_map = np.zeros((h, w), dtype=np.float32)

    # Get depth values in the valid region
    valid_depths = depth[depth_mask]  # [N]

    if len(valid_depths) == 0:
        return percentile_map

    # Sort valid depths
    sorted_depths = np.sort(valid_depths)
    n_valid = len(sorted_depths)

    # Use searchsorted to find the rank of each depth value
    # 'right' means find the first position greater than the value
    ranks = np.searchsorted(sorted_depths, depth[depth_mask], side='right')

    # Compute percentile: rank / n * 100
    percentiles = (ranks / n_valid) * 100.0

    # Fill results
    percentile_map[depth_mask] = percentiles

    return percentile_map


def calculate_camera_distance(cam1_extrinsic, cam2_extrinsic):
    """Calculate distance between two camera poses using translation and rotation

    Args:
        cam1_extrinsic: Camera 1 extrinsic matrix [B, 4, 4] or [4, 4]
        cam2_extrinsic: Camera 2 extrinsic matrix [B, 4, 4] or [4, 4]

    Returns:
        total_dist: [B] or scalar - Combined translation and rotation distance
    """
    # Handle both batched and single input
    is_batched = cam1_extrinsic.dim() == 3
    if not is_batched:
        cam1_extrinsic = cam1_extrinsic.unsqueeze(0)
        cam2_extrinsic = cam2_extrinsic.unsqueeze(0)

    # Extract translation vectors
    t1 = cam1_extrinsic[:, :3, 3]  # [B, 3]
    t2 = cam2_extrinsic[:, :3, 3]  # [B, 3]

    # Calculate translation distance
    translation_dist = torch.norm(t1 - t2, dim=1)  # [B]

    # Extract rotation matrices
    R1 = cam1_extrinsic[:, :3, :3]  # [B, 3, 3]
    R2 = cam2_extrinsic[:, :3, :3]  # [B, 3, 3]

    # Calculate rotation distance using Frobenius norm
    rotation_dist = torch.norm(R1 - R2, p='fro', dim=(1, 2))  # [B]

    # Combine translation and rotation distances
    total_dist = translation_dist + 0.1 * rotation_dist  # [B]

    if not is_batched:
        total_dist = total_dist.item()

    return total_dist


def get_camera_frustum_corners(K, extrinsic, image_width, image_height, depth_range=(0.1, 100.0)):
    """Get the 8 corners of camera frustum in world coordinates

    Args:
        K: Intrinsic matrix [B, 3, 3] or [3, 3]
        extrinsic: Extrinsic matrix (world to camera) [B, 4, 4] or [4, 4]
        image_width: Image width (scalar or tensor [B])
        image_height: Image height (scalar or tensor [B])
        depth_range: (near, far) depth range

    Returns:
        corners: [B, 8, 3] or [8, 3] tensor of frustum corners in world coordinates
    """
    near, far = depth_range

    # Handle both batched and single input
    is_batched = K.dim() == 3
    if not is_batched:
        K = K.unsqueeze(0)
        extrinsic = extrinsic.unsqueeze(0)

    batch_size = K.shape[0]
    device = K.device

    # Get camera center and rotation
    c2w = torch.inverse(extrinsic)  # [B, 4, 4] Camera to world
    camera_center = c2w[:, :3, 3]  # [B, 3]
    R = c2w[:, :3, :3]  # [B, 3, 3]

    # Get inverse intrinsic
    K_inv = torch.inverse(K)  # [B, 3, 3]

    # Handle scalar or tensor image dimensions
    if not isinstance(image_width, torch.Tensor):
        image_width = torch.tensor([image_width] * batch_size, device=device, dtype=torch.float32)
    if not isinstance(image_height, torch.Tensor):
        image_height = torch.tensor([image_height] * batch_size, device=device, dtype=torch.float32)

    # Define image corners in pixel coordinates for all batches
    # corners_2d: [B, 4, 3]
    corners_2d = torch.stack([
        torch.stack([torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)], dim=1),
        torch.stack([image_width, torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)], dim=1),
        torch.stack([image_width, image_height, torch.ones(batch_size, device=device)], dim=1),
        torch.stack([torch.zeros(batch_size, device=device), image_height, torch.ones(batch_size, device=device)], dim=1)
    ], dim=1)  # [B, 4, 3]

    # Unproject to normalized camera coordinates
    # K_inv: [B, 3, 3], corners_2d: [B, 4, 3]
    ray_dirs_cam = torch.bmm(corners_2d, K_inv.transpose(1, 2))  # [B, 4, 3]

    # Normalize ray directions
    ray_dirs_cam = ray_dirs_cam / torch.norm(ray_dirs_cam, dim=2, keepdim=True)  # [B, 4, 3]

    # Scale to near and far depths
    corners_cam_near = ray_dirs_cam * near  # [B, 4, 3]
    corners_cam_far = ray_dirs_cam * far  # [B, 4, 3]

    # Transform to world coordinates
    # R: [B, 3, 3], corners_cam_near: [B, 4, 3]
    corners_world_near = torch.bmm(corners_cam_near, R.transpose(1, 2)) + camera_center.unsqueeze(1)  # [B, 4, 3]
    corners_world_far = torch.bmm(corners_cam_far, R.transpose(1, 2)) + camera_center.unsqueeze(1)  # [B, 4, 3]

    # Combine near and far corners
    corners_world = torch.cat([corners_world_near, corners_world_far], dim=1)  # [B, 8, 3]

    if not is_batched:
        corners_world = corners_world.squeeze(0)  # [8, 3]

    return corners_world


def calculate_frustum_volume_overlap(corners1, corners2):
    """Calculate approximate overlap between two frustums using their corner points

    Args:
        corners1: [B, 8, 3] or [8, 3] tensor of frustum 1 corners
        corners2: [B, 8, 3] or [8, 3] tensor of frustum 2 corners

    Returns:
        overlap_score: [B] or scalar - Approximate overlap score (0 to 1)
    """
    # Handle both batched and single input
    is_batched = corners1.dim() == 3
    if not is_batched:
        corners1 = corners1.unsqueeze(0)
        corners2 = corners2.unsqueeze(0)

    # Calculate bounding boxes for both frustums
    min1 = torch.min(corners1, dim=1)[0]  # [B, 3]
    max1 = torch.max(corners1, dim=1)[0]  # [B, 3]
    min2 = torch.min(corners2, dim=1)[0]  # [B, 3]
    max2 = torch.max(corners2, dim=1)[0]  # [B, 3]

    # Calculate intersection of bounding boxes
    intersection_min = torch.max(min1, min2)  # [B, 3]
    intersection_max = torch.min(max1, max2)  # [B, 3]

    # Check if there is an intersection
    intersection_valid = torch.all(intersection_max > intersection_min, dim=1)  # [B]

    # Calculate volumes
    intersection_dims = torch.clamp(intersection_max - intersection_min, min=0.0)  # [B, 3]
    intersection_volume = torch.prod(intersection_dims, dim=1)  # [B]

    volume1 = torch.prod(max1 - min1, dim=1)  # [B]
    volume2 = torch.prod(max2 - min2, dim=1)  # [B]

    # Calculate overlap ratio (intersection over union)
    union_volume = volume1 + volume2 - intersection_volume  # [B]
    overlap_score = intersection_volume / (union_volume + 1e-8)  # [B]

    # Set overlap to 0 where there's no valid intersection
    overlap_score = torch.where(intersection_valid, overlap_score, torch.zeros_like(overlap_score))

    if not is_batched:
        overlap_score = overlap_score.item()

    return overlap_score


def calculate_fov_overlap(cam1_intrinsic, cam1_extrinsic, cam2_intrinsic, cam2_extrinsic, image_width, image_height, near, far):
    """Calculate FOV overlap between two cameras using frustum intersection

    This function constructs frustums from near and far planes and calculates their overlap.
    The overlap ratio represents the proportion of cam1's view that is also visible in cam2.
    Supports batched computation for efficiency.

    Args:
        cam1_intrinsic: Intrinsic matrix of camera 1 [B, 3, 3] or [3, 3]
        cam1_extrinsic: Extrinsic matrix of camera 1 [B, 4, 4] or [4, 4]
        cam2_intrinsic: Intrinsic matrix of camera 2 [B, 3, 3] or [3, 3]
        cam2_extrinsic: Extrinsic matrix of camera 2 [B, 4, 4] or [4, 4]
        image_width: Image width (scalar or tensor [B])
        image_height: Image height (scalar or tensor [B])

    Returns:
        overlap_ratio: [B] or scalar - Ratio of overlapping view (0 to 1)
        angle_between_cameras: [B] or scalar - Angle between camera viewing directions (degrees)
    """
    # Handle both batched and single input
    is_batched = cam1_intrinsic.dim() == 3
    if not is_batched:
        cam1_intrinsic = cam1_intrinsic.unsqueeze(0)
        cam1_extrinsic = cam1_extrinsic.unsqueeze(0)
        cam2_intrinsic = cam2_intrinsic.unsqueeze(0)
        cam2_extrinsic = cam2_extrinsic.unsqueeze(0)

    # Get camera centers and viewing directions
    c2w1 = torch.inverse(cam1_extrinsic)  # [B, 4, 4]
    c2w2 = torch.inverse(cam2_extrinsic)  # [B, 4, 4]

    cam1_center = c2w1[:, :3, 3]  # [B, 3]
    cam2_center = c2w2[:, :3, 3]  # [B, 3]

    # Camera viewing direction is the negative z-axis in camera space
    cam1_view_dir = c2w1[:, :3, 2]  # [B, 3] Third column of rotation matrix
    cam2_view_dir = c2w2[:, :3, 2]  # [B, 3]

    # Calculate angle between viewing directions
    cos_angle = torch.sum(cam1_view_dir * cam2_view_dir, dim=1)  # [B]
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle_between = torch.rad2deg(torch.acos(cos_angle))  # [B]

    # Construct frustums using near and far planes
    depth_range = (near, far)  # near and far planes

    # Get frustum corners for both cameras (batched)
    corners1 = get_camera_frustum_corners(
        cam1_intrinsic, cam1_extrinsic, image_width, image_height, depth_range
    )  # [B, 8, 3]
    corners2 = get_camera_frustum_corners(
        cam2_intrinsic, cam2_extrinsic, image_width, image_height, depth_range
    )  # [B, 8, 3]

    # Calculate frustum overlap (batched)
    overlap_ratio = calculate_frustum_volume_overlap(corners1, corners2)  # [B]

    if not is_batched:
        overlap_ratio = overlap_ratio.item() if isinstance(overlap_ratio, torch.Tensor) else overlap_ratio
        angle_between = angle_between.item()

    return overlap_ratio, angle_between


def find_closest_camera_in_view(target_extrinsic, ref_extrinsics, target_intrinsic, ref_intrinsics,
                                image_width, image_height, method="distance", near=0.1, far=5.0, angle_penalty=False,
                                shortcut_index=None, topk_return=0):
    """Find the camera in reference views that is closest to the target camera

    Args:
        target_extrinsic: Target camera extrinsic matrix [4, 4]
        ref_extrinsics: Reference camera extrinsic matrices [N, 4, 4]
        target_intrinsic: Target camera intrinsic matrix [3, 3]
        ref_intrinsics: Reference camera intrinsic matrices [N, 3, 3]
        image_width: Image width
        image_height: Image height
        method: "distance" for pose-based matching, "fov_overlap" for FOV overlap-based matching

    Returns:
        best_idx: Index of the closest camera in ref_extrinsics
        best_score: Score of the best match
    """
    num_refs = ref_extrinsics.shape[0]

    if num_refs == 0:
        return None, float('inf') if method == "distance" else -1.0

    # Expand target to match batch size
    target_extrinsic_batch = target_extrinsic.unsqueeze(0).expand(num_refs, -1, -1)  # [N, 4, 4]

    if method == "distance":
        # Batch calculate distances
        distances = calculate_camera_distance(target_extrinsic_batch, ref_extrinsics)  # [N]

        # Find the camera with minimum distance
        min_distance_idx = torch.argmin(distances)
        min_distance = distances[min_distance_idx].item()

        return min_distance_idx.item(), min_distance

    elif method == "fov_overlap":
        # Expand target intrinsic to match batch size
        target_intrinsic_batch = target_intrinsic.unsqueeze(0).expand(num_refs, -1, -1)  # [N, 3, 3]

        # Batch calculate FOV overlap
        overlap_ratios, angle_betweens = calculate_fov_overlap(
            target_intrinsic_batch, target_extrinsic_batch,
            ref_intrinsics, ref_extrinsics,
            image_width, image_height,
            near=near, far=far
        )  # overlap_ratios: [N], angle_betweens: [N]

        angle_betweens[angle_betweens < 0] = -angle_betweens[angle_betweens < 0]
        angle_betweens[angle_betweens > 180] = 360 - angle_betweens[angle_betweens > 180]

        if angle_penalty:
            # Penalize cameras with large angle difference
            overlap_ratios = overlap_ratios * torch.clip(torch.exp(((-angle_betweens + 90) / 180.0) * 5.0), 0.0, 1.0)

        if shortcut_index is not None:
            overlap_ratios[shortcut_index] += 1.0

        if topk_return == 0:
            # Find the camera with maximum overlap
            max_overlap_idx = torch.argmax(overlap_ratios)
            max_overlap = overlap_ratios[max_overlap_idx].item()
            angle_between = angle_betweens[max_overlap_idx].item()

            return max_overlap_idx.item(), max_overlap, angle_between
        else:
            max_overlap_indices = torch.topk(overlap_ratios, topk_return, dim=0, sorted=True, largest=True).indices
            return max_overlap_indices.tolist()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'distance' or 'fov_overlap'")


class CameraSelector:
    """Camera selector that fuses extrinsic parameters and image features to pick representative frames."""

    def __init__(
            self,
            feature_extractor='dinov2',
            device: str = 'cuda'
    ):
        self.device = device
        self.feature_extractor = feature_extractor
        self.model = None
        self.transform = None

        self._load_model(feature_extractor)

    def _load_model(self, extractor: str):
        """Load pre-trained feature extractor model."""
        if extractor == 'dinov2':
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        else:
            raise ValueError(f"Unknown feature extractor: {extractor}")

    @torch.no_grad()
    def extract_image_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract image features.

        Args:
            images: List of (H, W, 3) RGB images

        Returns:
            features: (N, D) feature vectors
        """

        features = []
        for img in images:
            # RGB -> PIL
            img_pil = Image.fromarray(img)

            # Extract
            if self.feature_extractor == 'dinov2':
                with torch.no_grad():
                    inputs = self.processor(images=img_pil, return_tensors="pt")
                    inputs.pixel_values = inputs.pixel_values.to(self.device)
                    feat = self.model(pixel_values=inputs.pixel_values).pooler_output  # (1, 768)
            else:
                raise ValueError(f"Unknown feature extractor: {self.feature_extractor}")

            features.append(feat.cpu().numpy().flatten())

        return np.array(features)

    def compute_image_quality_scores(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Compute image quality / information-richness scores.
        Used to bias FPS sampling toward higher-quality images.
        """
        scores = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Information entropy
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum() + 1e-10
            entropy = -np.sum(hist * np.log2(hist))

            # Contrast
            contrast = gray.std()

            # Combined score
            score = 0.4 * np.log1p(sharpness) + 0.3 * entropy + 0.3 * (contrast / 128.0)
            scores.append(score)

        return np.array(scores)

    def select(
            self,
            extrinsics: np.ndarray,
            images: List[np.ndarray],
            topk: int,
            camera_weight: float = 0.3,
            image_weight: float = 0.7,
            quality_bias: float = 0.1,  # quality preference weight
    ) -> Tuple[np.ndarray, dict]:
        """
        Select representative cameras by combining extrinsic parameters and image features.

        Args:
            extrinsics: (N, 4, 4) world-to-camera matrices
            images: List of (H, W, 3) RGB images
            topk: number of cameras to select
            camera_weight: weight for camera features
            image_weight: weight for image features
            quality_bias: image quality preference (larger = stronger bias toward high-quality images)

        Returns:
            indices: selected indices
            info: additional information dict
        """
        N = len(images)
        images = [im[0] for im in images]
        assert extrinsics.shape[0] == N
        assert topk <= N

        # 1. Extract camera features
        camera_features, positions = self._extract_camera_features(extrinsics)

        # 2. Extract image features
        image_features = self.extract_image_features(images)

        # 3. Compute image quality scores
        quality_scores = self.compute_image_quality_scores(images)

        # 4. Normalize
        camera_features = self._normalize(camera_features)
        image_features = self._normalize(image_features)

        # 5. Fuse features
        combined_features = np.concatenate([
            camera_features * camera_weight,
            image_features * image_weight
        ], axis=1)

        # 6. Quality-aware FPS
        indices = self._quality_aware_fps(combined_features, quality_scores, topk, quality_bias)

        info = {
            'positions': positions,
            'quality_scores': quality_scores,
            'selected_quality_scores': quality_scores[indices],
        }

        return indices, info

    def _extract_camera_features(self, extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract camera features (position + orientation)."""
        N = extrinsics.shape[0]
        positions = []
        orientations = []

        for i in range(N):
            w2c = extrinsics[i]
            R_mat = w2c[:3, :3]
            t = w2c[:3, 3]

            cam_pos = -R_mat.T @ t
            positions.append(cam_pos)

            quat = R.from_matrix(R_mat).as_quat()
            if quat[3] < 0:
                quat = -quat
            orientations.append(quat)

        positions = np.array(positions)
        orientations = np.array(orientations)
        features = np.concatenate([positions, orientations], axis=1)

        return features, positions

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        return (features - mean) / std

    def _quality_aware_fps(
            self,
            features: np.ndarray,
            quality_scores: np.ndarray,
            k: int,
            quality_bias: float
    ) -> np.ndarray:
        """
        Quality-aware farthest point sampling.
        Adds a quality penalty to distances so that low-quality images are less likely to be selected.
        """
        N = features.shape[0]
        selected = []
        distances = np.full(N, np.inf)

        # Normalize quality scores
        q_min, q_max = quality_scores.min(), quality_scores.max()
        q_normalized = (quality_scores - q_min) / (q_max - q_min + 1e-8)

        # Quality adjustment factor: low-quality images have a smaller "effective distance"
        quality_factor = 1.0 + quality_bias * (q_normalized - 0.5)

        # Start from the point closest to the centroid with highest quality
        centroid = features.mean(axis=0)
        centroid_dist = np.linalg.norm(features - centroid, axis=1)
        start_score = -centroid_dist + quality_bias * q_normalized
        current = np.argmax(start_score)
        selected.append(current)

        for _ in range(k - 1):
            dist_to_current = np.linalg.norm(features - features[current], axis=1)
            distances = np.minimum(distances, dist_to_current)

            # Adjusted distances (high-quality images have a larger effective distance)
            adjusted_distances = distances * quality_factor

            # Mark already-selected points as -inf
            adjusted_distances[selected] = -np.inf

            current = np.argmax(adjusted_distances)
            selected.append(current)

        return np.array(selected)


def voxel_downsample_fixed(points, colors, voxel_size):
    """
    Voxel downsampling with a fixed voxel_size (pure NumPy hash implementation, no open3d dependency).
    Takes the first point falling into each voxel as the representative, avoiding the overhead of
    open3d object construction/destruction and data format conversion.

    Args:
        points: numpy array [N, 3]
        voxel_size: float, voxel size
        colors: numpy array [N, 3] or None, point colors (uint8 or float)

    Returns:
        If colors is None: ds_points numpy array [M, 3]
        If colors is not None: (ds_points [M, 3], ds_colors [M, 3])
    """
    # Discretize point coordinates into integer voxel coordinates, then deduplicate with hashing
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    # Encode 3D voxel coordinates into a single int64 hash value
    hash_keys = voxel_coords[:, 0] * 116101979 + voxel_coords[:, 1] * 104729 + voxel_coords[:, 2]
    # Find the index of the first point in each unique voxel as the representative
    _, unique_indices = np.unique(hash_keys, return_index=True)
    ds_points = points[unique_indices]
    if colors is not None:
        return ds_points, colors[unique_indices]
    return ds_points


def adaptive_voxel_downsample(points, colors=None, N_points=1_000_000, tol=0.2, max_iter=10):
    """
    Adaptive voxel downsampling: uses a hash-accelerated binary search to find a voxel_size
    that brings the downsampled point count close to N_points.
    The search phase uses spatial hashing to quickly count unique voxels; once the best
    voxel_size is determined, voxel_downsample_fixed is called to perform the actual downsampling.

    Args:
        points: numpy array [N, 3], point cloud coordinates
        colors: numpy array [N, 3] or None, point colors (uint8 or float)
        N_points: target point count
        tol: allowed relative error (default 20%)
        max_iter: maximum number of binary search iterations

    Returns:
        ds_points: downsampled point cloud coordinates [M, 3]
        ds_colors: downsampled colors [M, 3] (if colors input is not None)
    """

    n_total = points.shape[0]
    if n_total <= N_points:
        return (points, colors, 0.003) if colors is not None else (points, None, 0.003)

    # Compute bbox extent in pure numpy to determine the upper bound for binary search
    bbox_extent = points.max(axis=0) - points.min(axis=0)
    voxel_lo = 1e-4
    voxel_hi = float(bbox_extent.max()) / 100

    best_voxel_size = (voxel_lo + voxel_hi) / 2.0
    for _ in range(max_iter):
        voxel_mid = (voxel_lo + voxel_hi) / 2.0
        # Hash counting: discretize point coordinates, deduplicate with numpy unique
        voxel_coords = np.floor(points / voxel_mid).astype(np.int64)
        # Encode 3D voxel coordinates into a single int64 hash to speed up unique
        hash_keys = voxel_coords[:, 0] * 116101979 + voxel_coords[:, 1] * 104729 + voxel_coords[:, 2]
        n_voxels = np.unique(hash_keys).shape[0]

        if abs(n_voxels - N_points) / N_points <= tol:
            best_voxel_size = voxel_mid
            break

        if n_voxels > N_points:
            voxel_lo = voxel_mid
        else:
            voxel_hi = voxel_mid
        best_voxel_size = voxel_mid

    # Call voxel_downsample_fixed to obtain the actual downsampled result
    result = voxel_downsample_fixed(points, colors, best_voxel_size)
    if colors is not None:
        ds_points, ds_colors = result
    else:
        ds_points = result
        ds_colors = None
    rank0_log(f"Voxel downsample: {n_total} -> {ds_points.shape[0]} points (target: {N_points}, voxel_size: {best_voxel_size:.6f})")

    return ds_points, ds_colors, best_voxel_size


class SimpleMemoryBank:
    def __init__(self, cfg, root_path, image_width, image_height, device, max_reference=8, align_nframe=8,
                 rank=0, world_size=1, camera_selector="dinov2"):
        # loading panorama info
        self.root_path = root_path
        self.image_width = image_width
        self.image_height = image_height

        self.nframe = cfg.nframe
        self.align_nframe = align_nframe  # number of frames (uniformly sampled) used for alignment
        self.max_reference = max_reference
        self.camera_selector = CameraSelector(camera_selector, device=device)

        # build panoramic memory bank
        memory_bank_path = f"{root_path}/pano_bank"
        if os.path.exists(memory_bank_path):
            memory_cameras = json.load(open(f"{memory_bank_path}/cameras.json"))

            # We share the same memory bank — preload everything into memory without saving intermediate results
            ref_image_list = glob(f"{memory_bank_path}/images/*.png")
            ref_image_list = sorted(ref_image_list)

            tasks = [(i, memory_cameras) for i in ref_image_list]

            def _load_item(args):
                img_p, cam_dict = args
                key = img_p.split('/')[-1].split('.')[0]
                view_id, traj_id = img_p.split('/')[-4], img_p.split('/')[-3]
                fname = f"{view_id}/{traj_id}/{key}"
                return (
                    np.array(cam_dict[key]['extrinsic']),
                    np.array(cam_dict[key]['intrinsic']),
                    Image.open(img_p).convert('RGB'),
                    fname
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                # map preserves result order consistent with tasks order
                results = list(tqdm(executor.map(_load_item, tasks), total=len(tasks), desc="Loading memory bank...", disable=rank != 0))

            if results:
                self.ref_w2cs, self.ref_Ks, self.ref_frames, self.fnames = map(list, zip(*results))
            else:
                self.ref_w2cs, self.ref_Ks, self.ref_frames, self.fnames = [], [], [], []
        else: # init the one-view simple memory bank
            start_frame = Image.open(f"{root_path}/image.png")
            origin_w, origin_h = start_frame.size
            height, width = assign_scale(origin_h, origin_w, scale_map=cfg.scale_map)
            start_frame = start_frame.resize((width, height), Image.Resampling.BICUBIC)
            self.ref_frames = [start_frame]
            self.fnames = ["renders/pano_bank/image.png"]
            self.ref_w2cs = [np.eye(4)]
            self.ref_Ks = [np.array(json.load(open(f"{root_path}/renders/traj_eloop/camera.json"))['intrinsic'][0])]

        self.ref_w2cs = torch.from_numpy(np.stack(self.ref_w2cs, axis=0)).to(dtype=torch.float32, device=device)
        self.ref_Ks = torch.from_numpy(np.stack(self.ref_Ks, axis=0)).to(dtype=torch.float32, device=device)
        self.mem_size = len(self.ref_frames)
        self.align_start_index = self.mem_size
        rank0_log(f"Initialized panorama memory size: {self.mem_size}")

        self.device = device
        self.rank = rank
        self.world_size = world_size

    def retrieval(self, tar_w2cs_full, tar_Ks_full, view_id=None, traj_id=None):
        """
        Return: retrieved_frames, ref_index, ref_index_dict
        """

        # If this is an aerial tracking trajectory, force-use a subset of the previous generation as retrieval results
        if ("wonder" in view_id or "target" in view_id) and traj_id > "traj0":
            shortcut_type = "aerial"
            shortcut_indices = [i for i, fname in enumerate(self.fnames) if fname.startswith(f"{view_id}/traj0")]
            rank0_log(f"Using {len(shortcut_indices)} history from {view_id}/traj0 ...")
            shortcut_indices = shortcut_indices[3::2]
        elif "view" in view_id and traj_id in ("traj0", "traj1"):
            shortcut_type = "regular"
            shortcut_indices = [i for i, fname in enumerate(self.fnames) if fname.startswith(f"{view_id}/traj2")]
            rank0_log(f"Using {len(shortcut_indices)} history from {view_id}/traj2 ...")
            if traj_id == "traj0":
                shortcut_indices = shortcut_indices[3::2]
            else:
                shortcut_indices = shortcut_indices[:len(shortcut_indices) // 2][3::2]
        else:
            shortcut_type = "none"
            shortcut_indices = []

        if tar_w2cs_full.shape[0] > self.nframe:
            tar_w2cs = tar_w2cs_full[0::4]
            tar_Ks = tar_Ks_full[0::4]
        else:
            tar_w2cs = tar_w2cs_full
            tar_Ks = tar_Ks_full

        # Track ref_index based on best_idx changes
        retrieval_map = dict()
        ref_index_dict = collections.defaultdict(dict)
        retrieved_frames = []
        retrieved_w2cs = []
        retrieved_Ks = []
        ref_w2cs = []
        retrieved_global_indices = []  # global list indices corresponding to retrieval results

        for i in range(1, tar_w2cs.shape[0]):
            if len(shortcut_indices) > 0:
                if shortcut_type == "aerial" and i not in (4, 8, 12, 16):
                    shortcut_index = shortcut_indices[int(i / tar_w2cs.shape[0] * len(shortcut_indices))]
                elif shortcut_type == "regular" and i in (1, 2, 3, 5, 7, 9, 13, 17):
                    shortcut_index = shortcut_indices[int(i / tar_w2cs.shape[0] * len(shortcut_indices))]
                else:
                    shortcut_index = None
            else:
                shortcut_index = None

            best_idx, best_score, angle_diff = find_closest_camera_in_view(
                tar_w2cs[i],
                self.ref_w2cs,
                tar_Ks[i],
                self.ref_Ks,
                self.image_width,
                self.image_height,
                method="fov_overlap",
                near=0.1,
                far=10.0,
                angle_penalty=True,
                shortcut_index=shortcut_index,
            )

            # Track ref_index when best_idx changes from previous frame
            if best_idx not in retrieval_map:
                retrieval_map[best_idx] = i - 1  # key: best_idx, value: index in retrieved_frames for this best_idx (only records the first occurrence)
            ref_index_dict[retrieval_map[best_idx]][i] = {"score": best_score, "angle_diff": angle_diff}  # key: index of retrieval frame in all retrieved_frames, value: corresponding target position

            retrieved_frames.append(np.array(self.ref_frames[best_idx])[None])
            retrieved_w2cs.append(self.ref_w2cs[best_idx])
            retrieved_Ks.append(self.ref_Ks[best_idx])
            retrieved_global_indices.append(best_idx)

        if len(ref_index_dict) > self.max_reference:
            rank0_log(f"Too many references. {len(ref_index_dict)} > {self.max_reference}")
            retrieved_w2cs = torch.stack(retrieved_w2cs)
            retrieved_Ks = torch.stack(retrieved_Ks)
            indices, _ = self.camera_selector.select(retrieved_w2cs.cpu().numpy(), retrieved_frames, topk=self.max_reference,
                                                     camera_weight=0.3, image_weight=0.7, quality_bias=0.1)
            retrieved_w2cs = retrieved_w2cs[indices]
            retrieved_Ks = retrieved_Ks[indices]
            retrieved_frames_selected = [retrieved_frames[i] for i in indices]
            retrieved_frames = []
            retrieved_global_indices_selected = [retrieved_global_indices[i] for i in indices]
            retrieved_global_indices = []

            # Re-assign retrieval results with the reduced candidate set
            retrieval_map = dict()
            ref_index_dict = collections.defaultdict(dict)

            for i in range(1, tar_w2cs.shape[0]):
                best_idx, best_score, angle_diff = find_closest_camera_in_view(
                    tar_w2cs[i],
                    retrieved_w2cs,
                    tar_Ks[i],
                    retrieved_Ks,
                    self.image_width,
                    self.image_height,
                    method="fov_overlap",
                    near=0.1,
                    far=10.0,
                    angle_penalty=True
                )

                # Track ref_index when best_idx changes from previous frame
                if best_idx not in retrieval_map:
                    retrieval_map[best_idx] = i - 1  # key: best_idx, value: index in retrieved_frames (first occurrence only)
                ref_index_dict[retrieval_map[best_idx]][i] = {"score": best_score, "angle_diff": angle_diff}  # key: retrieval frame index in all retrieved_frames, value: corresponding target position

                retrieved_frames.append(np.array(retrieved_frames_selected[best_idx]))
                ref_w2cs.append(retrieved_w2cs[best_idx])  # reorder retrieved_w2cs
                retrieved_global_indices.append(retrieved_global_indices_selected[best_idx])
        else:
            ref_w2cs = retrieved_w2cs  # copy retrieved_w2cs

        ref_index_list = list(ref_index_dict.keys())

        retrieved_frames = np.concatenate(retrieved_frames, axis=0)
        ref_index = torch.tensor(ref_index_list, dtype=torch.long)
        ref_w2cs = torch.stack(ref_w2cs)[ref_index]

        return retrieved_frames, ref_index, ref_index_dict, ref_w2cs, retrieved_global_indices

    def update_memory(self, gen_frames, tar_w2cs_full, tar_Ks_full, view_id=None, traj_id=None):
        """
        Update memory images in-memory only (all processes handle simultaneously).
        No alignment or IO operations are performed here.

        Args:
            gen_frames: [PIL.Image] * N — generated video frames
            tar_w2cs_full: full target world-to-camera matrices
            tar_Ks_full: full target intrinsic matrices
        """
        assert tar_w2cs_full.shape[0] == tar_Ks_full.shape[0] == len(gen_frames), f"{tar_w2cs_full.shape[0]} != {tar_Ks_full.shape[0]} != {len(gen_frames)}"

        # Subsample frames to be updated in the memory bank (skip the first frame)
        nframe = len(gen_frames)
        indices = sample_align_nframe(nframe, self.align_nframe)
        updated_tar_w2cs = tar_w2cs_full[indices]
        updated_tar_Ks = tar_Ks_full[indices]
        gen_frames = [gen_frames[idx] for idx in indices]

        # Update memory bank cache
        self.ref_w2cs = torch.cat([self.ref_w2cs, updated_tar_w2cs.to(self.device)], dim=0)
        self.ref_Ks = torch.cat([self.ref_Ks, updated_tar_Ks.to(self.device)], dim=0)
        self.ref_frames.extend(gen_frames)

        for index in indices:
            self.fnames.append(f"{view_id}/{traj_id}/{str(index).zfill(4)}")

    def apply_worldmirror(self, output_path, skip_exist=True):
        # Convert output to WorldMirror input format
        self.world_mirror_dir = output_path
        scene_name = output_path.split('/')[-3]

        if not (skip_exist and os.path.exists(f"{self.world_mirror_dir}/cameras.json")):
            os.makedirs(f"{self.world_mirror_dir}/images", exist_ok=True)

            # Distribute work across ranks: each rank processes its assigned frames
            process_list = np.arange(len(self.fnames))[self.rank::self.world_size]

            # Camera dict for this rank
            local_camera_dict = {"extrinsics": [], "intrinsics": []}
            # Collect image save tasks (image, save_path)
            save_tasks = []
            for gi in process_list:
                fname = self.fnames[gi]
                view_id, traj_id, frame_id = fname.split("/")
                if traj_id == "pano_bank":
                    camera_id = f"pano-{frame_id}"
                else:
                    camera_id = f"{view_id}-{traj_id}-{frame_id}"

                local_camera_dict["extrinsics"].append({
                    "camera_id": camera_id,
                    "matrix": self.ref_w2cs[gi].inverse().cpu().numpy().tolist()
                })
                local_camera_dict["intrinsics"].append({
                    "camera_id": camera_id,
                    "matrix": self.ref_Ks[gi].cpu().numpy().tolist()
                })
                save_tasks.append((self.ref_frames[gi], f"{self.world_mirror_dir}/images/{camera_id}.png"))

            # Multi-threaded image saving for speed
            def _save_image(args):
                img, path = args
                img.save(path)

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(_save_image, save_tasks))

            color_print(f"[Rank{self.rank}] Saved {len(save_tasks)} world mirror images.", "info")

            # Synchronize camera_dict across all ranks
            all_camera_dicts = [None] * self.world_size
            dist.all_gather_object(all_camera_dicts, local_camera_dict)

            # Rank 0 merges and saves cameras.json
            if self.rank == 0:
                merged_camera_dict = {"num_cameras": 0, "extrinsics": [], "intrinsics": []}
                for rank_dict in all_camera_dicts:
                    if rank_dict is not None:
                        merged_camera_dict["extrinsics"].extend(rank_dict["extrinsics"])
                        merged_camera_dict["intrinsics"].extend(rank_dict["intrinsics"])
                merged_camera_dict["num_cameras"] = len(merged_camera_dict["extrinsics"])
                with open(f"{self.world_mirror_dir}/cameras.json", "w") as f:
                    json.dump(merged_camera_dict, f, indent=2)
                color_print(f"[Rank0] Saved cameras.json with {merged_camera_dict['num_cameras']} cameras.", "info")

            dist.barrier()

        self.name_map = {}
        merged_camera_dict = json.load(open(f"{self.world_mirror_dir}/cameras.json", "r"))
        world_mirror_cam_ids = []
        for cam in merged_camera_dict["extrinsics"]:
            world_mirror_cam_ids.append(cam["camera_id"])
        world_mirror_cam_ids.sort()
        for i in range(len(world_mirror_cam_ids)):
            camera_id = world_mirror_cam_ids[i]
            if camera_id.startswith("pano-"):
                view_id, traj_id, fname_id = scene_name, "pano_bank", camera_id.split("-")[1]
            else:
                view_id, traj_id, fname_id = camera_id.split("-")
            self.name_map[f"{view_id}/{traj_id}/{fname_id}"] = str(i).zfill(4)

        with open(f"{self.world_mirror_dir}/name_map.json", "w") as f:
            json.dump(self.name_map, f, indent=2)

        dist.barrier()