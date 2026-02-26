import torch
import torch.nn.functional as F


def cam2world(points: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: (B, N, 3) - Points in camera frame
        R_c2w:  (B, 3, 3) - Rotation matrix
        t_c2w:  (B, 3)    - Translation vector
    Output:
        (B, N, 3) - Points in world frame
    Formulation:
        P_world = P_cam @ R_c2w^T + t_c2w
    """
    return points @ R_c2w.transpose(-2, -1) + t_c2w.unsqueeze(1)


def world2cam(points: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: (B, N, 3) - Points in world frame
        R_c2w:  (B, 3, 3) - Rotation matrix (Camera to World)
        t_c2w:  (B, 3)    - Translation vector (Camera to World)
    Output:
        (B, N, 3) - Points in camera frame
    Formulation:
        P_cam = (P_world - t_c2w) @ R_c2w
        (Exploiting orthogonality: R_w2c = R_c2w^T. We avoid explicit transpose by associating subtraction first)
    """
    return (points - t_c2w.unsqueeze(1)) @ R_c2w


def cam2image(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: (B, N, 3) - Points in camera frame
        K:      (B, 3, 3) - Intrinsic matrix
    Output:
        (B, N, 2) - Normalized pixel coordinates
    Formulation:
        P_hom = P_cam @ K^T
        pixels = P_hom_xy / P_hom_z
    """
    points_hom = points @ K.transpose(-2, -1)
    depth = points_hom[..., 2:3] + 1e-8
    return points_hom[..., :2] / depth


def image2cam(points: torch.Tensor, K: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: (B, N, 2) - Pixel coordinates
        K:      (B, 3, 3) - Intrinsic matrix
        depth:  (B, N)    - Depth map
    Output:
        (B, N, 3) - Points in camera frame
    Formulation:
        P_hom = [u, v, 1]
        P_cam = depth * (P_hom @ K^-T)
    """
    #assert torch.all(depth > 0), "Depth values must be positive."
    
    points_hom = F.pad(points, (0, 1), "constant", 1.0)
    inv_K = torch.linalg.inv(K)
    
    vecs = points_hom @ inv_K.transpose(-2, -1)
    return vecs * depth.unsqueeze(-1)


def world2image(points: torch.Tensor, K: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: (B, N, 3) - Points in world frame
        K:      (B, 3, 3) - Intrinsic matrix
        R_c2w:  (B, 3, 3) - Rotation matrix
        t_c2w:  (B, 3)    - Translation vector
    Output:
        (B, N, 2) - Pixel coordinates
    Formulation:
        Utilizes matrix association to fuse rotation and intrinsics.
        M = R_c2w @ K^T
        P_proj = (P_world - t_c2w) @ M
        pixels = P_proj_xy / P_proj_z
    """
    transform_matrix = R_c2w @ K.transpose(-2, -1)
    
    points_centered = points - t_c2w.unsqueeze(1)
    points_hom = points_centered @ transform_matrix
    
    depth = points_hom[..., 2:3] + 1e-8
    return points_hom[..., :2] / depth


def image2world(points: torch.Tensor, K: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: (B, N, 2) - Pixel coordinates
        K:      (B, 3, 3) - Intrinsic matrix
        R_c2w:  (B, 3, 3) - Rotation matrix
        t_c2w:  (B, 3)    - Translation vector
        depth:  (B, N)    - Depth map
    Output:
        (B, N, 3) - Points in world frame
    Formulation:
        Utilizes matrix association to fuse intrinsics and rotation.
        M = (R_c2w @ K^-1)^T
        P_dir = [u, v, 1] @ M
        P_world = (P_dir * depth) + t_c2w
    """
    points_hom = F.pad(points, (0, 1), "constant", 1.0)
    
    inv_K = torch.linalg.inv(K)
    M = (R_c2w @ inv_K).transpose(-2, -1)
    
    vecs_world_dir = points_hom @ M
    return vecs_world_dir * depth.unsqueeze(-1) + t_c2w.unsqueeze(1)


def compose_transformation(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
        R1, R2: (B, 3, 3) - Rotation matrices
        t1, t2: (B, 3)    - Translation vectors
    Output:
        R_composed: (B, 3, 3)
        t_composed: (B, 3)
    Formulation:
        T2(T1(x)) = R2(R1 x + t1) + t2
        R_new = R2 @ R1
        t_new = R2 @ t1 + t2
    """
    R_composed = R2 @ R1
    t_composed = (R2 @ t1.unsqueeze(-1)).squeeze(-1) + t2
    return R_composed, t_composed


def get_camera_coverage(H: int, W: int, K: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """
    Input:
        H, W:  Integers - Image height and width
        K:     (B, 3, 3) - Intrinsic matrix
        depth: (B, 2)    - Depth for corners
    Output:
        (B, 2) - Absolute coverage distance in camera frame
    Formulation:
        Unprojects top-left (0,0) and bottom-right (W-1, H-1) corners.
        Computes L1 distance between x-coordinates.
        Uses image2cam logic optimized for 2 points without full tensor repeat.
    """
    device = K.device
    B = K.shape[0]
    
    corner_pixels = torch.tensor([
        [0.0, 0.0], 
        [float(W - 1), float(H - 1)]
    ], device=device, dtype=torch.float32) 
    
    corner_pixels = corner_pixels.unsqueeze(0).expand(B, -1, -1)
    
    corner_cam = image2cam(corner_pixels, K, depth)
    
    return torch.abs(corner_cam[:, 1, 0:2] - corner_cam[:, 0, 0:2])


def get_cam_direction(K: torch.Tensor, R_c2w: torch.Tensor, only_xy: bool=False, norm: bool=True) -> torch.Tensor:
    """
    Input:
        K:     (B, 3, 3) - Intrinsic matrix (Used for device reference/shape)
        R_c2w: (B, 3, 3) - Rotation matrix
    Output:
        (B, 3) or (B, 2) - View direction vector in world frame
    Formulation:
        The camera optical axis is the Z-axis [0, 0, 1] in camera frame.
        In world frame, this corresponds strictly to the 3rd column of R_c2w.
        No matrix multiplication is required, only slicing.
    """
    direc_world = R_c2w[:, :, 2]
    
    if only_xy:
        direc_world = direc_world[:, 0:2]
        
    if norm:
        direc_world = F.normalize(direc_world, dim=-1)
        
    return direc_world