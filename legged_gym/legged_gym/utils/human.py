import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def load_target_jt(device, filename, offset=None, rng=None):
    target_jt = np.load(f"data/{filename}", allow_pickle=True)
    if len(target_jt.shape) == 2: # not 1 or 3
        target_jt = target_jt[None, :]
    target_jt = [torch.tensor(item.astype(np.float32)[:rng]).to(device) for item in target_jt]
    size = torch.zeros(len(target_jt)).to(device)
    for i, item in enumerate(target_jt):
        size[i] = item.size(0)
    padded_tensor = pad_sequence(target_jt, batch_first=True)
    if offset is not None:
        padded_tensor += offset

    return padded_tensor, size

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1

def sample_wp(device, num_points, num_wp, ranges):
    '''sample waypoints, relative to the starting point'''
    # position
    l_positions = torch.randn(num_points, 3) # left wrist positions
    l_positions = l_positions / l_positions.norm(dim=-1, keepdim=True) * ranges.wrist_max_radius # within a sphere, [-radius, +radius]
    l_positions = l_positions[l_positions[:,0] > ranges.l_wrist_pos_x[0]] # keep the ones that x > ranges.l_wrist_pos_x[0]
    l_positions = l_positions[l_positions[:,0] < ranges.l_wrist_pos_x[1]] # keep the ones that x < ranges.l_wrist_pos_x[1]
    l_positions = l_positions[l_positions[:,1] > ranges.l_wrist_pos_y[0]] # keep the ones that y > ranges.l_wrist_pos_y[0]
    l_positions = l_positions[l_positions[:,1] < ranges.l_wrist_pos_y[1]] # keep the ones that y < ranges.l_wrist_pos_y[1]
    l_positions = l_positions[l_positions[:,2] > ranges.l_wrist_pos_z[0]] # keep the ones that z > ranges.l_wrist_pos_z[0]
    l_positions = l_positions[l_positions[:,2] < ranges.l_wrist_pos_z[1]] # keep the ones that z < ranges.l_wrist_pos_z[1]

    r_positions = torch.randn(num_points, 3) # right wrist positions
    r_positions = r_positions / r_positions.norm(dim=-1, keepdim=True) * ranges.wrist_max_radius # within a sphere, [-radius, +radius]
    r_positions = r_positions[r_positions[:,0] > ranges.r_wrist_pos_x[0]] # keep the ones that x > ranges.r_wrist_pos_x[0]
    r_positions = r_positions[r_positions[:,0] < ranges.r_wrist_pos_x[1]] # keep the ones that x < ranges.r_wrist_pos_x[1]
    r_positions = r_positions[r_positions[:,1] > ranges.r_wrist_pos_y[0]] # keep the ones that y > ranges.r_wrist_pos_y[0]
    r_positions = r_positions[r_positions[:,1] < ranges.r_wrist_pos_y[1]] # keep the ones that y < ranges.r_wrist_pos_y[1]
    r_positions = r_positions[r_positions[:,2] > ranges.r_wrist_pos_z[0]] # keep the ones that z > ranges.r_wrist_pos_z[0]
    r_positions = r_positions[r_positions[:,2] < ranges.r_wrist_pos_z[1]] # keep the ones that z < ranges.r_wrist_pos_z[1]
    
    num_pairs = min(l_positions.size(0), r_positions.size(0))
    positions = torch.stack([l_positions[:num_pairs], r_positions[:num_pairs]], dim=1) # (num_pairs, 2, 3)
    
    # rotation (quaternion)
    quaternions = torch.randn(num_pairs, 2, 4)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    
    # concat
    wp = torch.cat([positions, quaternions], dim=-1) # (num_pairs, 2, 7)
    # repeat for num_wp
    wp = wp.unsqueeze(1).repeat(1, num_wp, 1, 1) # (num_pairs, num_wp, 2, 7)
    print("===> [sample_wp] return shape:", wp.shape)
    return wp.to(device), num_pairs, num_wp


def sample_rp(device, num_points, num_wp, ranges):
    """sample reach points"""
    wp, num_pairs, num_wp = sample_wp(device, num_points, num_wp, ranges)
    center_positions = (torch.rand(num_pairs, 3) * ranges.max_center_distance).to(device)
    center_positions[:, 0] = torch.clamp(center_positions[:, 0], ranges.center_offset_x[0], ranges.center_offset_x[1])
    center_positions[:, 1] = torch.clamp(center_positions[:, 1], ranges.center_offset_y[0], ranges.center_offset_y[1])
    center_positions[:, 2] = torch.clamp(center_positions[:, 2], ranges.center_offset_z[0], ranges.center_offset_z[1])
    center_positions = center_positions.unsqueeze(1).repeat(1, num_wp, 1) # (num_pairs, num_wp, 3)
    center_positions = center_positions.unsqueeze(2).repeat(1, 1, 2, 1)
    wp[:, :, :, :3] += center_positions
    print("===> [sample_rp] return shape:", wp.shape)
    return wp.to(device), num_pairs, num_wp


def sample_fp(device, num_points, num_wp, ranges):
    '''sample feet waypoints'''
    # left foot still, right foot move, [num_points//2, 2]
    l_positions_s = torch.zeros(num_points//2, 2) # left foot positions (xy)
    r_positions_m = torch.randn(num_points//2, 2)
    r_positions_m = r_positions_m / r_positions_m.norm(dim=-1, keepdim=True) * ranges.feet_max_radius # within a sphere, [-radius, +radius]
    # right foot still, left foot move, [num_points//2, 2]
    r_positions_s = torch.zeros(num_points//2, 2) # right foot positions (xy)
    l_positions_m = torch.randn(num_points//2, 2)
    l_positions_m = l_positions_m / l_positions_m.norm(dim=-1, keepdim=True) * ranges.feet_max_radius # within a sphere, [-radius, +radius]
    # concat
    l_positions = torch.cat([l_positions_s, l_positions_m], dim=0) # (num_points, 2)
    r_positions = torch.cat([r_positions_m, r_positions_s], dim=0) # (num_points, 2)
    wp = torch.stack([l_positions, r_positions], dim=1) # (num_points, 2, 2)
    wp = wp.unsqueeze(1).repeat(1, num_wp, 1, 1) # (num_points, num_wp, 2, 2)
    print("===> [sample_fp] return shape:", wp.shape)
    return wp.to(device), num_points, num_wp
    

def sample_root_height(device, num_points, num_wp, ranges, base_height_target):
    '''sample root height'''
    root_height = torch.randn(num_points, 1) * ranges.root_height_std + base_height_target
    root_height = torch.clamp(root_height, ranges.min_root_height, ranges.max_root_height)
    root_height = root_height.unsqueeze(1).repeat(1, num_wp, 1) # (num_points, num_wp, 1)
    print("===> [sample_root_height] return shape:", root_height.shape)
    return root_height.to(device), num_points, num_wp