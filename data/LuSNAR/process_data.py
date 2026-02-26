import os
import pandas as pd
import torch
from typing import Optional, List, Dict
from scipy.spatial.transform import Rotation
import numpy as np

def quat2R(quat: torch.Tensor):
    """
    Convert quaternion to rotation matrix.
    Args:        
        quat (np.ndarray): Quaternion in the form [w, x, y, z].
    Returns:     
        np.ndarray: Rotation matrix corresponding to the quaternion.
    """
    # (w,x,y,z) -> (x,y,z,w)
    quat = quat.numpy()
    quat = np.roll(quat, -1)
    R = Rotation.from_quat(quat).as_matrix()
    R = torch.from_numpy(R).float()
    return R

def compose_transformation(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched composition of two transformations: R1, t1 followed by R2, t2.
    The resulting transformation takes a point p and computes: R2 @ (R1 @ p + t1) + t2

    Args:
        R1 (torch.Tensor): First rotation matrix, shape (B, 3, 3).
        t1 (torch.Tensor): First translation vector, shape (B, 3).
        R2 (torch.Tensor): Second rotation matrix, shape (B, 3, 3).
        t2 (torch.Tensor): Second translation vector, shape (B, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Composed rotation and translation.
    """
    R_composed = R2 @ R1
    # Unsqueeze t1 for matmul, then add t2.
    t_composed = (R2 @ t1.unsqueeze(-1)).squeeze(-1) + t2
    return R_composed, t_composed

K = torch.tensor([[610.17784, 0, 512], [0, 610.17784, 512], [0, 0, 1]], dtype=torch.float32)
quat_left2rov = torch.tensor([0.579, 0.406, 0.406, 0.579])
R_left2rov = quat2R(quat_left2rov)
t_left2rov = torch.tensor([1.000, -0.155, -1.500])

def read_pose_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads pose data from a CSV file, removes duplicates, and sorts by timestamp.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Pose file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, dtype={'timestamp': 'int64'})
        
        df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        return df.sort_values(by='timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None

def read_pairs_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads rover-satellite timestamp pairs from the 'pairs.txt' file.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Pairs file not found at {file_path}")
        return None
    try:
        # The file has a simple structure: rover_timestamp,sat_timestamp
        df = pd.read_csv(file_path, dtype={'rover_timestamp': 'int64', 'sat_timestamp': 'int64'})
        return df
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None

def create_pair_splits(
    split_scenes: Dict[str, List[int]],
    base_path: str = './raw_data',
    output_path: str = './'
):
    """
    Processes the generated data and creates training, validation, and test splits.

    Args:
        train_scenes: A list of scene IDs (e.g., [1, 2, 3, 4]) for the training set.
        val_scenes: A list of scene IDs for the validation set.
        test_scenes: A list of scene IDs for the test set.
        base_path: The root directory where the raw_data is stored.
        output_path: The directory where the final .pth files will be saved.
    """
    print("\n--- Starting Data Split Creation ---")
    os.makedirs(output_path, exist_ok=True)

    # Process each split
    for split_name, scene_ids in split_scenes.items():
        if not scene_ids:
            print(f"Skipping {split_name} set (no scenes provided).")
            continue

        print(f"Processing {split_name} set for scenes: {scene_ids}...")
        
        all_data_pairs = []
        for scene_id in scene_ids:
            scene_name = f"Moon_{scene_id}"
            scene_path = os.path.join(base_path, scene_name)
            
            # Load the pose data for rover (left) and satellite images
            pairs_df = read_pairs_data(os.path.join(scene_path, 'pairs.txt'))
            if scene_id < 10:
                rover_poses_df = read_pose_data(os.path.join(scene_path, 'rover_poses_interpolated.txt'))
            else:
                rover_poses_df = read_pose_data(os.path.join(scene_path, 'left_camera_poses.txt'))
            sat_poses_df = read_pose_data(os.path.join(scene_path, 'sat_poses.txt'))

            # Check if all required data is present
            if pairs_df is None or rover_poses_df is None or sat_poses_df is None:
                print(f"  Warning: Skipping scene {scene_id} for {split_name} split due to missing data file(s).")
                continue

            # --- Use timestamp as index for fast lookups ---
            rover_poses_df.set_index('timestamp', inplace=True)
            sat_poses_df.set_index('timestamp', inplace=True)
            
            for _, pair in pairs_df.iterrows():
                rover_timestamp = int(pair['rover_timestamp'])
                sat_timestamp = int(pair['sat_timestamp'])

                try:
                    rover_row = rover_poses_df.loc[rover_timestamp]
                    sat_row = sat_poses_df.loc[sat_timestamp]
                except KeyError as e:
                    print(f"  Warning: Timestamp {e} not found in pose files for scene {scene_id}. Skipping pair.")
                    continue

                quat_rov = torch.tensor([rover_row.qw, rover_row.qx, rover_row.qy, rover_row.qz])
                t_rov2world = torch.tensor([rover_row.x, rover_row.y, rover_row.z], dtype=torch.float32)
                R_rov2world = quat2R(quat_rov)
                
                global R_left2rov, t_left2rov, K
                R_left2world, t_left2world = compose_transformation(R_left2rov.unsqueeze(0), 
                                                                    t_left2rov.unsqueeze(0), 
                                                                    R_rov2world.unsqueeze(0), 
                                                                    t_rov2world.unsqueeze(0))
                R_left2world, t_left2world = R_left2world.squeeze(0), t_left2world.squeeze(0)
                
                # --- Create Satellite Data ---
                quat_sat = torch.tensor([sat_row.qw, sat_row.qx, sat_row.qy, sat_row.qz])
                t_sat = torch.tensor([sat_row.x, sat_row.y, sat_row.z], dtype=torch.float32)

                K_left = K
                K_sat = K
                data_dict = {
                    "left_image_path": os.path.join('LuSNAR', scene_path, 'left_images', f"{rover_timestamp}.png"),
                    "K_left": K_left,
                    "R_left2world": R_left2world,
                    "t_left2world": t_left2world,
                    "x_offset_ratio": pair['x_offset_ratio'],
                    "y_offset_ratio": pair['y_offset_ratio'],

                    "sat_image_path": os.path.join('LuSNAR', scene_path, 'sat_images', f"{sat_timestamp}.png"),
                    "K_sat": K_sat, # Assuming same intrinsics
                    "R_sat2world": quat2R(quat_sat),
                    "t_sat2world": t_sat,
                    "left_depth_path": os.path.join('LuSNAR', scene_path, 'depths', f"{rover_timestamp}.pfm"),
                }
                all_data_pairs.append(data_dict)
        
        # Save the final list to a .pth file
        if all_data_pairs:
            output_file = os.path.join(output_path, f"{split_name}_data.pth")
            torch.save(all_data_pairs, output_file)
            print(f"Successfully saved {split_name} set with {len(all_data_pairs)} samples to {output_file}")

    print("--- Finished Data Split Creation ---")


if __name__ == '__main__':

    create_pair_splits({'train': [1,2,3,4,5,6,9],
                        'val': [7],
                        'test': [8]},
                        output_path='./')

