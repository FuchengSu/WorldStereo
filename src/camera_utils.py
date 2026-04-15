# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Visualization utilities using trimesh
# --------------------------------------------------------
import PIL.Image
import scipy
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from pytorch3d.renderer.cameras import look_at_rotation
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from scipy.interpolate import splprep, splev
import trimesh

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]

TRAJECTORY_COLORS = {
        "regular_0": {"start": (255, 70, 70), "end": (255, 140, 100)},  # red -> light red
        "regular_1": {"start": (50, 200, 80), "end": (120, 255, 150)},  # green -> light green
        "regular_2": {"start": (50, 100, 255), "end": (100, 180, 255)},  # blue -> light blue

        "surround": {"start": (255, 180, 30), "end": (255, 220, 80)},  # golden yellow
        "reconstruct": {"start": (30, 200, 220), "end": (100, 255, 220)},  # cyan
        "exploration": {"start": (200, 80, 255), "end": (255, 150, 255)},  # purple -> pink
        "aerial": {"start": (30, 144, 255), "end": (80, 220, 255)},  # blue
    }


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([p[:3, 3] for p in poses]))


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None,
                  screen_width=0.03, marker=None, edge_width=1.0):
    """
    edge_width: border thickness multiplier, default 1.0 (larger = thicker, recommended range 1.0~5.0)
    """
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H, W) * 1.1

    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5 ** 0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    base_offset = 0.05        # original offset = 1 - 0.95
    base_angle  = 2.0         # original rotation angle (degrees)

    scale_factor = 1.0 - base_offset * edge_width   # edge_width=1 → 0.95
    rot_angle    = base_angle * edge_width           # edge_width=1 → 2°

    # Clamp scale_factor to avoid it becoming too small
    scale_factor = max(scale_factor, 0.5)

    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(rot_angle)).as_matrix()

    vertices = np.r_[cam.vertices,
                      scale_factor * cam.vertices,
                      geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2 * len(cam.vertices)

        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width / 4)
        marker.vertices += pose_c2w[:3, 3]
        marker.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(marker)


def camera_backward_forward(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([0, 0, distance, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def camera_left_right(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([distance, 0, 0, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def native_camera_rotation(c2w, medium_depth, phi, theta):
    R_elevation = np.array([[1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1]], dtype=np.float32)
    R_azimuth = np.array([[np.cos(phi), 0, np.sin(phi), 0],
                          [0, 1, 0, 0],
                          [-np.sin(phi), 0, np.cos(phi), 0],
                          [0, 0, 0, 1]], dtype=np.float32)

    dummy_c2w = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -medium_depth],
                          [0, 0, 0, 1]], dtype=np.float32)
    dummy_c2w = R_azimuth @ R_elevation @ dummy_c2w
    dummy_c2w[:3, 3] += np.array([0, 0, medium_depth], dtype=np.float32)
    c2w = c2w @ dummy_c2w

    return c2w


def axis_angle_to_matrix(axis, angle):
    """Rodrigues rotation formula, axis must be unit vector, angle in radians"""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ])
    return R


def make_homogeneous_rotation(axis, angle, point):
    """
    Construct a 4x4 homogeneous transformation for rotation about an arbitrary axis
    that does not pass through the origin.

    Args:
        axis: (3,) rotation axis direction (normalized)
        angle: scalar, angle in radians
        point: (3,) a point on the rotation axis

    Returns:
        4x4 transformation matrix
    """
    # Rotation part
    R = axis_angle_to_matrix(axis, angle)
    # Translate to origin
    T1 = np.eye(4)
    T1[:3, 3] = -point
    # Rotate
    R_homo = np.eye(4)
    R_homo[:3, :3] = R
    # Translate back
    T2 = np.eye(4)
    T2[:3, 3] = point
    # Combined transform
    M = T2 @ R_homo @ T1
    return M


def rotate_cam2world_around_axis(cam2world, axis, angle, point):
    """
    cam2world: 4x4 numpy array
    axis: 3-element array, rotation axis direction (arbitrary direction in space, not necessarily normalized)
    angle: float, rotation angle (radian)
    point: 3-element array, a point on the rotation axis
    """
    axis = np.asarray(axis)
    point = np.asarray(point)
    M = make_homogeneous_rotation(axis, angle, point)
    # Apply M @ cam2world or cam2world @ M depending on whether operating in world or camera space
    new_cam2world = M @ cam2world
    return new_cam2world


def camera_rotation(c2w, medium_depth, phi, theta):
    # For iterative camera motion, first compute the initial pitch angle
    z0 = c2w[2, 3]
    if z0 != 0 and phi != 0:
        axis_origin = np.array([0, 0, medium_depth, 1], dtype=np.float32)
        axis_origin = c2w @ axis_origin.reshape(4, 1)
        axis_origin = axis_origin[:3, 0]
        axis_origin[2] = 0
        return rotate_cam2world_around_axis(c2w, axis=np.array([0, 0, 1], dtype=np.float32), angle=-phi, point=axis_origin)
    else:
        return native_camera_rotation(c2w, medium_depth, phi, theta)


def interpolate_poses(poses, M):
    """
    Interpolate camera extrinsics.

    Args:
        poses: (N, 4, 4) numpy array, N camera extrinsic matrices
        M: number of cameras after interpolation, M > N

    Returns:
        (M, 4, 4) numpy array of interpolated extrinsics
    """
    N = poses.shape[0]
    assert N >= 2, "At least two poses are required for interpolation"
    assert poses.shape[1:] == (4, 4), "Pose format error: expected (N, 4, 4)"

    # Time parameter: uniformly spaced in [0, 1]
    t_orig = np.linspace(0, 1, N)
    t_interp = np.linspace(0, 1, M)

    # 1) Extract rotations and translations
    rotations = poses[:, :3, :3]  # (N,3,3)
    translations = poses[:, :3, 3]  # (N,3)

    # 2) Convert rotations to quaternions
    r = scipy.spatial.transform.Rotation.from_matrix(rotations)

    # 3) Create Slerp object
    slerp = scipy.spatial.transform.Slerp(t_orig, r)

    # 4) Interpolate rotations
    interp_rots = slerp(t_interp)
    interp_rot_mats = interp_rots.as_matrix()  # (M,3,3)

    # 5) Linearly interpolate translations
    interp_trans = np.empty((M, 3))
    for i in range(3):
        interp_trans[:, i] = np.interp(t_interp, t_orig, translations[:, i])

    # 6) Assemble final matrices
    interp_poses = np.zeros((M, 4, 4))
    interp_poses[:, :3, :3] = interp_rot_mats
    interp_poses[:, :3, 3] = interp_trans
    interp_poses[:, 3, 3] = 1.0
    return interp_poses


def compute_points_to_mesh_distance(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Compute the minimum distance from N points to a mesh.

    Args:
        points: (N, 3) numpy array
        mesh: open3d TriangleMesh

    Returns:
        distances: (N,) numpy array, minimum distance from each point to the mesh
    """
    # Convert mesh to tensor format
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # Compute unsigned distances (closest surface distance)
    query_points = o3d.core.Tensor(points.astype(np.float32))
    distances = scene.compute_distance(query_points)

    return distances.numpy()


def get_c2w(c2w_start, move, median_depth, air_bound, n_inter=20, kdtree=None, mesh=None, distance_threshold=0.02, local_rank=0, obs_decay=0.5, obs_limit=4):
    distance_threshold = min(distance_threshold, median_depth * 0.1)
    if type(distance_threshold) == torch.Tensor:
        distance_threshold = distance_threshold.item()

    c2ws = []
    if move["type"] == "normal":
        for j in range(n_inter):
            # Construct intermediate camera motion
            move_inter = dict()
            for key in move:
                if key not in ("type", "name"):
                    if type(move[key]) == list:
                        move_inter[key] = [(j + 1) / n_inter * v for v in move[key]]
                    else:
                        move_inter[key] = (j + 1) / n_inter * move[key]

            c2w = c2w_start.copy()
            for key in move_inter:
                if key == "rotation" and np.sum(np.abs(move_inter[key])) == 0:
                    continue
                if key in ("backward-forward", "left-right") and move_inter[key] == 0:
                    continue

                if key == "backward-forward":
                    c2w = camera_backward_forward(c2w, air_bound * move_inter[key])
                elif key == "left-right":
                    c2w = camera_left_right(c2w, air_bound * move_inter[key])
                elif key == "rotation":
                    phi, theta = move_inter[key]
                    phi = np.deg2rad(phi)
                    theta = np.deg2rad(theta)
                    c2w = camera_rotation(c2w, median_depth, phi, theta)
                else:
                    raise NotImplementedError
            c2ws.append(c2w)
    elif move["type"] == "eloop":
        look_at_point = (0, 0, median_depth)
        angles = np.linspace(0, 2 * np.pi, n_inter + 1)[1:]
        move["radius_x"] *= median_depth
        move["radius_y"] *= median_depth
        for angle in angles:
            cam_pos = np.array([move['radius_x'] * np.sin(angle), move['radius_y'] * np.cos(angle) - move['radius_y'], 0], dtype=np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = cam_pos
            R_new = look_at_rotation(cam_pos, at=(look_at_point,), up=((0, 1, 0),), device="cpu").numpy()[0]
            c2w[:3, :3] = R_new
            c2w = c2w_start @ c2w
            c2ws.append(c2w)
    elif move["type"] == "aerial":
        phi, theta = move["rotation"]
        n_inter_phi = int(round(np.abs(phi) / (np.abs(phi) + np.abs(theta)) * n_inter))
        n_inter_theta = n_inter - n_inter_phi
        for j in range(n_inter_theta):  # first rotate in elevation (pitch), then horizontal (aerial)
            theta_j = np.deg2rad((j + 1) / n_inter_theta * theta)
            c2w = camera_rotation(c2w_start.copy(), median_depth, 0, theta_j)
            c2ws.append(c2w)
        c2w_middle = c2ws[-1].copy()
        for j in range(n_inter_phi):
            phi_j = np.deg2rad((j + 1) / n_inter_phi * phi)
            c2w = camera_rotation(c2w_middle.copy(), median_depth, phi_j, 0)
            c2ws.append(c2w)
    else:
        raise NotImplementedError

    c2ws = np.array(c2ws)
    if c2ws.shape[0] < 80:
        c2ws_ = interpolate_poses(c2ws, 80)
    else:
        c2ws_ = c2ws.copy()
    # Query nearest neighbor indices and distances
    query_points = c2ws_[:, :3, 3]
    if mesh is not None:  # prefer mesh for obstacle avoidance
        distances = compute_points_to_mesh_distance(query_points, mesh)
    else:
        distances, indices = kdtree.query(query_points, k=5)
        distances = distances.mean(axis=1)
    min_distance = distances.min()
    obs_iteration = 0
    while min_distance < distance_threshold and obs_iteration < obs_limit + 1:  # reduce travel distance by half, at most obs_limit times
        if local_rank == 0:
            print(f"Obstruction is detected in candidate: {move}", "min distance:", min_distance, "reduce the trajectory by half...")
        if move["type"] == "normal":
            c2ws = c2ws[:int(c2ws.shape[0] * obs_decay)]  # obs_decay: reduce to obs_decay fraction of the original trajectory range
            c2ws = interpolate_poses(c2ws, n_inter)
        elif move["type"] == "eloop":
            c2ws = []
            look_at_point = (0, 0, median_depth)
            angles = np.linspace(0, 2 * np.pi, n_inter + 1)[1:]
            move["radius_x"] *= obs_decay
            move["radius_y"] *= obs_decay
            for angle in angles:
                cam_pos = np.array([move['radius_x'] * np.sin(angle), move['radius_y'] * np.cos(angle) - move['radius_y'], 0], dtype=np.float32)
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, 3] = cam_pos
                R_new = look_at_rotation(cam_pos, at=(look_at_point,), up=((0, 1, 0),), device="cpu").numpy()[0]
                c2w[:3, :3] = R_new
                c2w = c2w_start @ c2w
                c2ws.append(c2w)
            c2ws = np.array(c2ws)
        elif move["type"] == "aerial":
            c2ws = []
            phi, theta = move["rotation"]
            n_dist_phi = int(round(np.abs(phi) / (np.abs(phi) + np.abs(theta)) * distances.shape[0]))
            n_dist_theta = distances.shape[0] - n_dist_phi
            dist_ = distances < distance_threshold
            if np.sum(dist_[:n_dist_theta]) > 0:  # elevation collision: reduce pitch angle
                move["rotation"][1] *= obs_decay
            if np.sum(dist_[n_dist_theta:]) > 0:  # horizontal collision: reduce azimuth angle
                move["rotation"][0] *= obs_decay
            phi, theta = move["rotation"]
            n_inter_phi = int(round(np.abs(phi) / (np.abs(phi) + np.abs(theta)) * n_inter))
            n_inter_theta = n_inter - n_inter_phi
            for j in range(n_inter_theta):  # first elevation rotation, then horizontal aerial rotation
                theta_j = np.deg2rad((j + 1) / n_inter_theta * theta)
                c2w = camera_rotation(c2w_start.copy(), median_depth, 0, theta_j)
                c2ws.append(c2w)
            c2w_middle = c2ws[-1].copy()
            for j in range(n_inter_phi):
                phi_j = np.deg2rad((j + 1) / n_inter_phi * phi)
                c2w = camera_rotation(c2w_middle.copy(), median_depth, phi_j, 0)
                c2ws.append(c2w)
            c2ws = np.array(c2ws)
        else:
            raise NotImplementedError
        if c2ws.shape[0] < 80:
            c2ws_ = interpolate_poses(c2ws, 80)
        else:
            c2ws_ = c2ws.copy()
        query_points = c2ws_[:, :3, 3]
        distances, indices = kdtree.query(query_points, k=5)
        distances = distances.mean(axis=1)
        min_distance = distances.min()
        obs_iteration += 1
        # print(f"New distance: {min_distance}")

    # print(f"candidate: {move}", f"min distance: {min_distance}", f"obs iteration: {obs_iteration}")

    return c2ws, obs_iteration


def sample_ones_from_binary_map(
        binary_map: np.ndarray,
        n_samples: int,
        random_seed: int = None,
) -> np.ndarray:
    """
    Sample n_samples locations **with replacement** from a binary numpy array at positions where value is 1.
    """
    # 1. Input validation
    if binary_map.ndim != 2:
        raise ValueError(f"Input must be a 2D array, got ndim={binary_map.ndim}")
    if not np.all(np.isin(binary_map, [0, 1])):
        raise ValueError("Input array must contain only 0 and 1")

    # 2. Extract coordinates of all positions where value is 1 (rows, cols)
    rows, cols = np.where(binary_map == 1)
    n_ones = len(rows)
    if n_ones == 0:
        raise ValueError("No positions with value 1 found; cannot sample")

    # 3. Set random seed (optional)
    if random_seed is not None:
        np.random.seed(random_seed)

    # 4. Sample with replacement (replace=True explicitly for clarity)
    sample_indices = np.random.choice(n_ones, size=n_samples, replace=True)

    # 5. Retrieve coordinates and combine into an (n_samples, 2) array
    sampled_points = np.stack([rows[sample_indices], cols[sample_indices]], axis=1)

    return sampled_points


def get_random_rotation_matrix(min_deg=45, max_deg=135, upright=False):
    """
    Generate a World-to-Camera rotation matrix (OpenCV coordinate system) satisfying angular constraints.

    Args:
        min_deg: minimum angle (degrees)
        max_deg: maximum angle (degrees)
        upright:
            False (default) -> fully random rotation (includes random roll)
            True -> force camera to be level (X-axis parallel to ground plane)

    Returns:
        R (np.ndarray): 3x3 rotation matrix
    """

    # 1. Construct camera Z axis (Forward)
    min_rad = np.deg2rad(min_deg)
    max_rad = np.deg2rad(max_deg)

    # Uniformly sample spherical coordinates
    cos_theta = np.random.uniform(np.cos(max_rad), np.cos(min_rad))
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi)

    z_c = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        cos_theta
    ])

    # 2. Construct camera X axis (Right)
    if upright:
        # Mode A: keep camera level
        world_up = np.array([0, 0, 1])
        x_c = np.cross(world_up, z_c)
        # Direction correction is handled automatically in step 4
        if np.linalg.norm(x_c) < 1e-6:
            # Rare case: z_c parallel to world_up (0° or 180°); handle for robustness
            x_c = np.array([1, 0, 0])
        else:
            x_c = x_c / np.linalg.norm(x_c)
    else:
        # Mode B: fully random roll
        random_vec = np.random.randn(3)
        x_c = random_vec - np.dot(random_vec, z_c) * z_c
        x_c = x_c / np.linalg.norm(x_c)

    # 3. Construct camera Y axis (Down)
    # By right-hand rule: Y = Z cross X
    y_c = np.cross(z_c, x_c)
    y_c = y_c / np.linalg.norm(y_c)

    # =================================================================
    # 4. Constraint check and correction
    # Requirement: camera Y axis (Down) must have an angle < 90° with the world -Z direction (World Down)
    # Equivalent to: Dot(y_c, [0,0,-1]) > 0  =>  -y_c[2] > 0  =>  y_c[2] < 0
    # =================================================================
    if y_c[2] >= 0:
        # If Y axis Z component >= 0, the camera is "upside down" or Y points upward.
        # Strategy: flip both X and Y simultaneously.
        # Rationale: (-X) cross (-Y) = Z, keeping Z unchanged and right-handed,
        # while rotating the camera 180° around Z so Y points downward.
        x_c = -x_c
        y_c = -y_c

    # 5. Assemble rotation matrix
    # Rows of R_cw are the camera axes
    R_cw = np.stack([x_c, y_c, z_c])

    return R_cw


def get_origin_height(bottom_mesh):
    """
    Get the bottom height of a mesh.

    Args:
        bottom_mesh: open3d.geometry.TriangleMesh (Legacy mesh)

    Returns:
        origin_height (float): bottom height of the mesh.
    """
    bottom_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(bottom_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(bottom_mesh_t)

    ray_origin = np.zeros((1, 3), dtype=np.float32)
    # Direction: (0, 0, -1) pointing straight down
    ray_dir = np.zeros((1, 3), dtype=np.float32)
    ray_dir[:, 2] = -1.0

    # Build ray tensor: shape (N, 6) -> [ox, oy, oz, dx, dy, dz]
    ray = np.concatenate([ray_origin, ray_dir], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)

    # Cast rays
    # cast_rays returns a dict with t_hit (distance), geometry_ids, etc.
    ans = scene.cast_rays(rays)

    # Parse results
    t_hit = ans['t_hit'].numpy()  # get distances

    z_values_bottom = ray_origin[:, 2] - t_hit

    return z_values_bottom


def get_z_from_xy(bottom_mesh, upper_mesh, x, y, z_max):
    """
    Given (x, y), cast rays downward to compute the z value on the mesh surface.

    Args:
        bottom_mesh: open3d.geometry.TriangleMesh (Legacy mesh)
        upper_mesh: open3d.geometry.TriangleMesh (Legacy mesh)
        x, y: [N,]*2

    Returns:
        z (float): z coordinate of the intersection. Returns NaN if no intersection.
    """

    # 1. Convert mesh to tensor format (required by RaycastingScene)
    # Skip this step if mesh is already o3d.t.geometry.TriangleMesh
    bottom_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(bottom_mesh)
    upper_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(upper_mesh)

    # 2. Create raycasting scene and add mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(bottom_mesh_t)

    # 3. Determine ray origins and directions
    # Ray origin: (x, y, z_max + 10.0) slightly above the mesh to ensure it starts outside

    # Origin: (x, y, z_max + 10.0) slightly above to ensure outside the object
    ray_origin = np.zeros((x.shape[0], 3), dtype=np.float32)
    ray_origin[:, 0] = x
    ray_origin[:, 1] = y
    ray_origin[:, 2] = z_max + 10.0
    # Direction: (0, 0, -1) pointing straight down
    ray_dir = np.zeros((x.shape[0], 3))
    ray_dir[:, 2] = -1.0

    # Build ray tensor: shape (N, 6) -> [ox, oy, oz, dx, dy, dz]
    ray = np.concatenate([ray_origin, ray_dir], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)

    # 4. Cast rays
    # cast_rays returns a dict with t_hit (distance), geometry_ids, etc.
    ans = scene.cast_rays(rays)

    # 5. Parse results
    t_hit = ans['t_hit'].numpy()  # get distances

    # Compute actual Z values
    # Z_intersection = Z_origin + (direction_z * t_hit)
    # Since direction_z = -1: Z_origin - t_hit
    z_values_bottom = ray_origin[:, 2] - t_hit
    # Set inf (no hit) to NaN
    z_values_bottom[np.isinf(t_hit)] = np.nan

    # Repeat for upper part
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(upper_mesh_t)

    ray_origin = np.zeros((x.shape[0], 3), dtype=np.float32)
    ray_origin[:, 0] = x
    ray_origin[:, 1] = y
    ray_origin[:, 2] = z_values_bottom + 1e-3  # slightly above the bottom Z value
    # Direction: (0, 0, 1) pointing straight up
    ray_dir = np.zeros((x.shape[0], 3))
    ray_dir[:, 2] = 1.0

    # Build ray tensor: shape (N, 6) -> [ox, oy, oz, dx, dy, dz]
    ray = np.concatenate([ray_origin, ray_dir], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    t_hit = ans['t_hit'].numpy()
    z_values_upper = ray_origin[:, 2] + t_hit
    z_values_upper[np.isinf(t_hit)] = np.nan

    return z_values_bottom, z_values_upper


def add_camera_pose_noise(c2w, trans_noise_range, rot_noise_degree_range):
    """
    Add random perturbation to N*4*4 c2w matrices in the local camera coordinate system.

    Args:
        c2w: (N, 4, 4) numpy array, original camera extrinsics
        trans_noise_range: list/tuple of 3 floats [x_max, y_max, z_max]
                           translation perturbation range along each axis (meters/scene units);
                           sampled uniformly in [-max, max].
        rot_noise_degree_range: list/tuple of 3 floats [x_deg, y_deg, z_deg]
                                rotation perturbation range along each axis (degrees);
                                sampled uniformly in [-deg, deg].

    Returns:
        perturbed_c2w: (N, 4, 4) perturbed extrinsics
    """
    N = c2w.shape[0]

    # 1. Generate translation noise -> (N, 3); sampled in [-limit, limit]
    tx_noise = np.random.uniform(-trans_noise_range[0], trans_noise_range[0], N)
    ty_noise = np.random.uniform(-trans_noise_range[1], trans_noise_range[1], N)
    tz_noise = np.random.uniform(-trans_noise_range[2], trans_noise_range[2], N)
    t_noise = np.stack([tx_noise, ty_noise, tz_noise], axis=1)

    # 2. Generate rotation noise -> (N, 3, 3) via Euler angles to rotation matrix
    rx_noise = np.random.uniform(-rot_noise_degree_range[0], rot_noise_degree_range[0], N)
    ry_noise = np.random.uniform(-rot_noise_degree_range[1], rot_noise_degree_range[1], N)
    rz_noise = np.random.uniform(-rot_noise_degree_range[2], rot_noise_degree_range[2], N)
    euler_noise = np.stack([rx_noise, ry_noise, rz_noise], axis=1)

    # Use scipy for efficient conversion; 'xyz' is the rotation order, degrees=True means input is in degrees
    rot_mat_noise = R.from_euler('xyz', euler_noise, degrees=True).as_matrix()

    # 3. Build noise transform matrix T_noise (N, 4, 4)
    noise_mat = np.eye(4)[None, ...].repeat(N, axis=0)  # initialize as identity matrices
    noise_mat[:, :3, :3] = rot_mat_noise
    noise_mat[:, :3, 3] = t_noise

    # 4. Apply perturbation
    # c2w @ noise_mat applies perturbation in the camera local coordinate system (recommended)
    # noise_mat @ c2w applies perturbation in the world global coordinate system
    perturbed_c2w = c2w @ noise_mat

    return perturbed_c2w


def compute_lookat_xy_angle(c2ws_R: Union[torch.Tensor, np.ndarray]):
    """
    Compute the angle between each camera's look-at direction and the XY plane (OpenCV coordinate system).

    OpenCV coordinate system:
        x -> right
        y -> down
        z -> forward (camera viewing direction)

    Args:
        c2ws_R: (N, 3, 3) camera-to-world rotation matrices

    Returns:
        angles: (N,) angle between each camera and the XY plane (degrees), range [0, 90]
    """
    is_numpy = isinstance(c2ws_R, np.ndarray)
    if is_numpy:
        c2ws_R = torch.from_numpy(c2ws_R).float()

    # OpenCV: camera looks along +z, i.e. R @ [0, 0, 1]^T = 3rd column of R
    look_at = c2ws_R[:, :, 2]  # (N, 3)

    # Normalize
    look_at = look_at / torch.norm(look_at, dim=-1, keepdim=True)

    # Angle with XY plane = arcsin(|z component|)
    z_component = look_at[:, 2]
    angles_rad = torch.arcsin(torch.clamp(torch.abs(z_component), -1.0, 1.0))
    angles_deg = torch.rad2deg(angles_rad)

    if is_numpy:
        angles_deg = angles_deg.numpy()

    return angles_deg


def create_arrow_mesh(start, end, color, shaft_radius=0.003, head_radius=0.008, head_length_ratio=0.2):
    """
    Create an arrow mesh from start to end (cylinder shaft + cone head).

    Args:
        start:             (3,) start coordinate
        end:               (3,) end coordinate
        color:             (3,) or (4,) color
        shaft_radius:      shaft cylinder radius
        head_radius:       cone base radius
        head_length_ratio: fraction of total length used for the arrowhead

    Returns:
        trimesh.Trimesh: arrow mesh
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-8:
        return None

    direction_normalized = direction / length

    # Arrow part lengths
    head_length = length * head_length_ratio
    shaft_length = length - head_length

    # ---- Shaft (cylinder) ---- #
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius,
        height=shaft_length,
        sections=8,
    )
    # Default cylinder along Z axis, centered at origin -> shift toward start direction
    shaft.apply_translation([0, 0, shaft_length / 2])

    # ---- Head (cone) ---- #
    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=8,
    )
    # Default cone along Z axis, base at origin -> move to shaft tip
    head.apply_translation([0, 0, shaft_length + head_length / 2])

    # ---- Merge ---- #
    arrow = trimesh.util.concatenate([shaft, head])

    # ---- Rotate: align from Z axis to direction ---- #
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    cross = np.cross(z_axis, direction_normalized)
    dot = np.dot(z_axis, direction_normalized)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            rotation_matrix = np.eye(3)
        else:
            # 180° rotation
            rotation_matrix = np.diag([1, -1, -1]).astype(np.float64)
    else:
        cross_normalized = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1, 1))
        rotation_matrix = Rotation.from_rotvec(cross_normalized * angle).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = start
    arrow.apply_transform(transform)

    # ---- Colorize ---- #
    color = np.asarray(color, dtype=np.uint8)
    if len(color) == 3:
        color = np.append(color, 255)
    arrow.visual.face_colors = color

    return arrow


def add_trajectory_arrows(
    scene,
    poses_c2w,
    edge_color,
    arrow_interval=1,
    shaft_radius=0.003,
    head_radius=0.008,
    head_length_ratio=0.25,
    arrow_scale=1.0,
    show_trajectory_line=True,
    line_color=None,
):
    """
    Visualize camera trajectory movement direction (arrows) in a trimesh.Scene.

    Args:
        scene:               trimesh.Scene object
        poses_c2w:           list of (4,4) np.ndarray, c2w matrices in temporal order
        edge_color:          (3,) color (same as camera color)
        arrow_interval:      draw an arrow every N frames (1 = every frame)
        shaft_radius:        shaft thickness
        head_radius:         arrowhead size
        head_length_ratio:   arrowhead fraction of total length
        arrow_scale:         overall arrow scale (> 1 lengthen, < 1 shorten)
        show_trajectory_line: whether to also draw a trajectory line
        line_color:          trajectory line color, None = same as edge_color
    """
    if len(poses_c2w) < 2:
        return

    edge_color = np.asarray(edge_color, dtype=np.uint8)
    if line_color is None:
        line_color = edge_color

    # Extract all camera centers
    centers = np.array([pose[:3, 3] for pose in poses_c2w])  # (T, 3)

    # ---- 1. Trajectory line (thin cylinder mesh) ---- #
    if show_trajectory_line:
        line_radius = shaft_radius * 0.4
        for i in range(len(centers) - 1):
            seg_start = centers[i]
            seg_end = centers[i + 1]
            seg_len = np.linalg.norm(seg_end - seg_start)
            if seg_len < 1e-8:
                continue

            seg_dir = (seg_end - seg_start) / seg_len
            line_seg = trimesh.creation.cylinder(
                radius=line_radius,
                height=seg_len,
                sections=6,
            )

            # Align direction
            z_axis = np.array([0, 0, 1.0])
            cross = np.cross(z_axis, seg_dir)
            dot = np.dot(z_axis, seg_dir)
            if np.linalg.norm(cross) < 1e-8:
                rot = np.eye(3) if dot > 0 else np.diag([1, -1, -1.0])
            else:
                angle = np.arccos(np.clip(dot, -1, 1))
                rot = Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = (seg_start + seg_end) / 2
            line_seg.apply_transform(T)

            lc = np.append(line_color, 180) if len(line_color) == 3 else line_color
            line_seg.visual.face_colors = lc.astype(np.uint8)
            scene.add_geometry(line_seg)

    # ---- 2. Direction arrows ---- #
    for i in range(0, len(centers) - 1, arrow_interval):
        start = centers[i]
        end = centers[i + 1]

        # Optional: scale arrow length by arrow_scale
        if arrow_scale != 1.0:
            direction = end - start
            mid = (start + end) / 2
            start = mid - direction * arrow_scale / 2
            end = mid + direction * arrow_scale / 2

        arrow = create_arrow_mesh(
            start, end, edge_color,
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length_ratio=head_length_ratio,
        )
        if arrow is not None:
            scene.add_geometry(arrow)


# ============================================================
#  1. Smooth trajectory curve (B-Spline interpolation)
# ============================================================
def smooth_trajectory(centers, num_points=300, smoothing=0.0):
    """
    Fit a B-Spline through discrete camera centers to obtain a smooth curve.

    Args:
        centers: (N, 3) camera center coordinates
        num_points: number of interpolated points (more = smoother)
        smoothing: smoothing factor; 0 = exact pass through all points, > 0 allows deviation

    Returns:
        smooth_pts: (num_points, 3)
    """
    if len(centers) < 4:
        # Too few points: fall back to linear interpolation
        t_orig = np.linspace(0, 1, len(centers))
        t_new = np.linspace(0, 1, num_points)
        smooth_pts = np.column_stack([
            np.interp(t_new, t_orig, centers[:, i]) for i in range(3)
        ])
        return smooth_pts

    tck, u = splprep([centers[:, 0], centers[:, 1], centers[:, 2]],
                     s=smoothing, k=min(3, len(centers) - 1))
    u_new = np.linspace(0, 1, num_points)
    smooth_pts = np.array(splev(u_new, tck)).T  # (num_points, 3)
    return smooth_pts


# ============================================================
#  2. Gradient-colored tube mesh along a curve
# ============================================================
def create_gradient_tube(points, radius=0.008, sections=12,
                         color_start=(30, 144, 255),
                         color_end=(0, 255, 200),
                         alpha=230):
    """
    Create a gradient-colored tube mesh along a series of points.

    Args:
        points: (N, 3) curve sample points
        radius: tube radius
        sections: number of polygon sides in tube cross-section
        color_start: start color RGB
        color_end: end color RGB
        alpha: opacity

    Returns:
        trimesh.Trimesh tube mesh
    """

    N = len(points)
    if N < 2:
        return None

    # Generate color gradient
    colors = np.zeros((N, 4), dtype=np.uint8)
    for i in range(N):
        t = i / max(N - 1, 1)
        colors[i, :3] = np.array(color_start) * (1 - t) + np.array(color_end) * t
        colors[i, 3] = alpha

    # Generate ring vertices for each cross-section
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(sections)])  # (S, 3)

    rings = []
    for i in range(N):
        # Compute local coordinate frame
        if i == 0:
            tangent = points[1] - points[0]
        elif i == N - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]

        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-10:
            tangent = np.array([0, 0, 1.0])
        else:
            tangent = tangent / tangent_len

        # Find a non-parallel vector to build the coordinate frame
        up = np.array([0, 1, 0]) if abs(np.dot(tangent, [0, 1, 0])) < 0.99 else np.array([1, 0, 0])
        right = np.cross(tangent, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, tangent)

        # Transform ring to world coordinates
        ring = circle[:, 0:1] * right + circle[:, 1:2] * up  # (S, 3)
        ring = ring * radius + points[i]
        rings.append(ring)

    rings = np.array(rings)  # (N, S, 3)

    # Build triangle faces
    vertices = rings.reshape(-1, 3)  # (N*S, 3)
    faces = []
    face_colors = []

    for i in range(N - 1):
        for j in range(sections):
            j_next = (j + 1) % sections
            v0 = i * sections + j
            v1 = i * sections + j_next
            v2 = (i + 1) * sections + j
            v3 = (i + 1) * sections + j_next

            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

            # Average color from both ends of the segment
            avg_color = ((colors[i].astype(float) + colors[i + 1].astype(float)) / 2).astype(np.uint8)
            face_colors.append(avg_color)
            face_colors.append(avg_color)

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.visual.face_colors = np.array(face_colors, dtype=np.uint8)
    return mesh


# ============================================================
#  3. Clean camera frustum for paper-quality visualization
# ============================================================
def add_clean_camera_frustum(scene, pose_c2w, color=(30, 144, 255),
                             frustum_size=0.06, aspect_ratio=16 / 9,
                             line_width=2.0, alpha=200):
    """
    Draw a clean camera frustum suitable for paper figures.
    Renders only 5 edges (4 side edges + rectangular base frame) without filled faces.

    Args:
        scene: trimesh.Scene
        pose_c2w: (4, 4) camera-to-world
        color: RGB
        frustum_size: frustum size
        aspect_ratio: width/height ratio
        line_width: line thickness (simulated with cylinders)
        alpha: opacity
    """
    # 5 vertices of the frustum (in camera coordinate system)
    h = frustum_size
    w = h * aspect_ratio * 0.5
    h_half = h * 0.5

    # Camera coordinates: z points forward (adjusted to OpenCV convention)
    apex = np.array([0, 0, 0])
    corners = np.array([
        [-w, -h_half, -frustum_size * 1.5],  # bottom-left
        [w, -h_half, -frustum_size * 1.5],   # bottom-right
        [w, h_half, -frustum_size * 1.5],    # top-right
        [-w, h_half, -frustum_size * 1.5],   # top-left
    ])

    # Transform to world coordinates
    OPENGL_TO_OPENCV = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    T = pose_c2w @ OPENGL_TO_OPENCV

    apex_w = (T @ np.append(apex, 1))[:3]
    corners_w = (T @ np.hstack([corners, np.ones((4, 1))]).T).T[:, :3]

    # Edges to draw
    edges = [
        (apex_w, corners_w[0]),
        (apex_w, corners_w[1]),
        (apex_w, corners_w[2]),
        (apex_w, corners_w[3]),
        (corners_w[0], corners_w[1]),
        (corners_w[1], corners_w[2]),
        (corners_w[2], corners_w[3]),
        (corners_w[3], corners_w[0]),
    ]

    cyl_radius = frustum_size * 0.008 * line_width
    color_rgba = np.array([*color, alpha], dtype=np.uint8)

    for start, end in edges:
        cyl = _create_cylinder_between(start, end, cyl_radius)
        if cyl is not None:
            cyl.visual.face_colors = color_rgba
            scene.add_geometry(cyl)


def _create_cylinder_between(p0, p1, radius, sections=8):
    """Create a cylinder between two points."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    diff = p1 - p0
    length = np.linalg.norm(diff)
    if length < 1e-10:
        return None

    direction = diff / length
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)

    # Align direction
    z = np.array([0, 0, 1.0])
    cross = np.cross(z, direction)
    dot = np.dot(z, direction)

    if np.linalg.norm(cross) < 1e-8:
        rot = np.eye(3) if dot > 0 else np.diag([1, -1, -1.0])
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        rot = Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = (p0 + p1) / 2
    cyl.apply_transform(T)
    return cyl


# ============================================================
#  4. Start/end markers (spheres + optional arrows)
# ============================================================
def add_endpoint_markers(scene, centers, color_start=(30, 144, 255),
                         color_end=(0, 255, 200), radius=0.02):
    """Add prominent sphere markers at the trajectory start and end points."""
    # Start - large sphere
    sphere_start = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere_start.vertices += centers[0]
    sphere_start.visual.face_colors = np.array([*color_start, 255], dtype=np.uint8)
    scene.add_geometry(sphere_start)

    # End - large sphere
    sphere_end = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere_end.vertices += centers[-1]
    sphere_end.visual.face_colors = np.array([*color_end, 255], dtype=np.uint8)
    scene.add_geometry(sphere_end)


def add_direction_arrow(scene, start, end, color=(0, 255, 200),
                        shaft_radius=0.006, head_radius=0.018,
                        head_length=0.03):
    """Add a direction arrow at the endpoint of a trajectory."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return

    direction = direction / length

    # Shaft
    shaft_length = length - head_length
    if shaft_length > 0:
        shaft_end = start + direction * shaft_length
        shaft = _create_cylinder_between(start, shaft_end, shaft_radius)
        if shaft is not None:
            shaft.visual.face_colors = np.array([*color, 230], dtype=np.uint8)
            scene.add_geometry(shaft)

    # Arrowhead (cone)
    cone = trimesh.creation.cone(radius=head_radius, height=head_length, sections=16)

    z = np.array([0, 0, 1.0])
    cross = np.cross(z, direction)
    dot = np.dot(z, direction)
    if np.linalg.norm(cross) < 1e-8:
        rot = np.eye(3) if dot > 0 else np.diag([1, -1, -1.0])
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        rot = Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = end - direction * head_length * 0.5
    cone.apply_transform(T)
    cone.visual.face_colors = np.array([*color, 255], dtype=np.uint8)
    scene.add_geometry(cone)


# ============================================================
#  5. Trajectory visualization
# ============================================================
def add_paper_trajectory(
        scene,
        c2ws,
        color_start=(30, 144, 255),  # start color (blue)
        color_end=(0, 255, 200),     # end color (cyan)
        tube_radius=0.008,           # tube thickness
        tube_sections=12,            # tube cross-section resolution
        tube_alpha=220,              # tube opacity
        smooth_points=300,           # number of smoothing interpolation points
        smooth_factor=0.0,           # smoothing coefficient
        show_cameras=True,           # whether to show camera frustums
        camera_interval=5,           # draw a camera every N frames
        frustum_size=0.06,           # frustum size
        frustum_line_width=1.5,      # frustum line width
        show_endpoint_markers=True,  # start/end sphere markers
        marker_radius=0.015,         # marker sphere size
        show_final_arrow=True,       # direction arrow at end point
        arrow_length_ratio=0.08,     # arrow length as fraction of total trajectory
):
    """
    All-in-one paper-quality camera trajectory visualization.

    Features:
      - B-Spline smoothed tube trajectory
      - Gradient color to indicate motion direction (no dense arrows needed)
      - Sparse + clean camera frustums
      - Start/end sphere markers
      - Direction arrow at the endpoint
    """
    if len(c2ws) < 2:
        return

    c2ws = c2ws[1:]

    centers = np.array([pose[:3, 3] for pose in c2ws])

    # ---- 1. Smooth tube trajectory ---- #
    smooth_pts = smooth_trajectory(centers, num_points=smooth_points,
                                   smoothing=smooth_factor)
    tube = create_gradient_tube(smooth_pts, radius=tube_radius,
                                sections=tube_sections,
                                color_start=color_start,
                                color_end=color_end,
                                alpha=tube_alpha)
    if tube is not None:
        scene.add_geometry(tube)

    # ---- 2. Sparse camera frustums ---- #
    if show_cameras:
        n = len(c2ws)
        indices = list(range(0, n, camera_interval))
        # Ensure start and end are included
        if 0 not in indices:
            indices = [0] + indices
        if (n - 1) not in indices:
            indices.append(n - 1)

        for idx in indices:
            t = idx / max(n - 1, 1)
            # Interpolate color along trajectory
            cam_color = tuple(
                int(color_start[c] * (1 - t) + color_end[c] * t)
                for c in range(3)
            )
            add_clean_camera_frustum(
                scene, c2ws[idx], color=cam_color,
                frustum_size=frustum_size,
                line_width=frustum_line_width,
            )

    # ---- 3. Start/end markers ---- #
    if show_endpoint_markers:
        add_endpoint_markers(scene, centers,
                             color_start=color_start,
                             color_end=color_end,
                             radius=marker_radius)

    # ---- 4. Direction arrow at endpoint ---- #
    if show_final_arrow and len(smooth_pts) >= 2:
        total_length = np.sum(np.linalg.norm(np.diff(smooth_pts, axis=0), axis=1))
        arrow_len = total_length * arrow_length_ratio

        # Take direction from the last segment
        tail = smooth_pts[-1]
        direction = smooth_pts[-1] - smooth_pts[-10]
        direction = direction / np.linalg.norm(direction)
        arrow_start = tail
        arrow_end = tail + direction * arrow_len

        add_direction_arrow(
            scene, arrow_start, arrow_end,
            color=color_end,
            shaft_radius=tube_radius * 0.8,
            head_radius=tube_radius * 3,
            head_length=arrow_len * 0.4,
        )