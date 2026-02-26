import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphGenerator(nn.Module):
    def __init__(self, num_nodes_per_scale, hid_dim, out_dim, num_sat_scales, strides=[1, 2, 4, 8]):
        """
        Args:
            num_nodes_per_scale (int): Number of keypoints to sample per scale (K).
            hid_dim (int): Feature dimension of the input maps.
            out_dim (int): Feature dimension of the output node features.
            num_sat_scales (int): Dimension of the output scale_weights vector.
            strides (list): The downsampling factors corresponding to the input feature maps.
        """
        super().__init__()

        self.num_nodes = num_nodes_per_scale
        self.strides = strides
        self.num_sat_scales = num_sat_scales

        self.reliability_head = nn.Sequential(
            nn.Conv2d(hid_dim, out_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_dim, 1, kernel_size=1, padding=0)
        )

        self.feature_head = nn.Sequential(
            nn.Conv2d(hid_dim, out_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)
        )

    def simple_nms(self, weights, nms_radius, mask_val='-inf'):
        """ Fast Non-maximum suppression to remove nearby points based on reliability weights """
        assert(nms_radius >= 0)
        
        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

        mask_val_float = {'zero': 0.0, '-inf': float('-inf')}[mask_val]
        mask_tensor = torch.ones_like(weights) * mask_val_float
        
        max_mask = weights == max_pool(weights)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_weights = torch.where(supp_mask, mask_tensor, weights)
            new_max_mask = supp_weights == max_pool(supp_weights)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, weights, mask_tensor)

    def mask_borders(self, weights, border, mask_val='-inf'):
        """ Masks out weights that are too close to the border """
        b, c, h, w = weights.shape
        mask = torch.zeros_like(weights)
        
        if h > 2*border and w > 2*border:
            mask[:, :, border:h-border, border:w-border] = 1
        else:
            mask[:, :, :, :] = 1 

        mask_val_float = {'zero': 0.0, '-inf': float('-inf')}[mask_val]
        mask_tensor = torch.ones_like(weights) * mask_val_float
        
        weights = torch.where(mask > 0, weights, mask_tensor)
        return weights

    def forward_left_featmaps(self, feature_maps):
        """
        Args:
            feature_maps (dict): Dictionary containing feature maps. 
                                 Keys should correspond to strides (e.g. 'scale_1', 'scale_16').
        Returns:
            all_norm_xy: [B, Total_Nodes, 2]
            all_reliability: [B, Total_Nodes] - The reliability weights used for sampling.
            all_features: [B, Total_Nodes, C]
            all_scale_weights: [B, Total_Nodes, num_sat_scales]
        """
        
        list_norm_xy = []
        list_reliability = []
        list_features = []

        for stride in self.strides:
            # Resolve key name
            key = f'scale_{stride}' 

            feature_head = self.feature_head
            reliability_head = self.reliability_head
            
            if key not in feature_maps:
                raise KeyError(f"Feature map for stride {stride} not found.")

            featmap = feature_maps[key]
            

            B, C, H, W = featmap.shape
            
            reliability_map = reliability_head(featmap) # [B, 1, H, W]
            
            proj_featmap = feature_head(featmap) # [B, C, H, W]

            border_size = max(1, H // 64) 
            nms_radius = max(1, H // 64)
            
            reliability_map = self.mask_borders(reliability_map, border=border_size, mask_val='-inf')
            reliability_map = self.simple_nms(reliability_map, nms_radius=nms_radius, mask_val='-inf')

            flat_reliability = reliability_map.flatten(2) # [B, 1, H*W]
            topk_reliability, topk_indices = torch.topk(flat_reliability, self.num_nodes, dim=2)
            
            y = topk_indices // W
            x = topk_indices % W
            
            norm_x = (x.float() / (W - 1)) * 2 - 1
            norm_y = (y.float() / (H - 1)) * 2 - 1
            norm_xy = torch.stack([norm_x, norm_y], dim=-1) # [B, 1, K, 2]
            
            grid = norm_xy # [B, 1, K, 2] for grid_sample
            
            sampled_feats = F.grid_sample(proj_featmap, grid, mode='bilinear', align_corners=False)
            sampled_feats = sampled_feats.squeeze(2).permute(0, 2, 1) # [B, K, C]

            list_norm_xy.append(norm_xy.squeeze(1))
            list_reliability.append(topk_reliability.squeeze(1))
            list_features.append(sampled_feats)

        all_norm_xy = torch.cat(list_norm_xy, dim=1)
        all_reliability = torch.cat(list_reliability, dim=1)
        all_features = torch.cat(list_features, dim=1)

        return all_norm_xy, all_reliability, all_features

    def forward_sat_featmaps(self, feature_maps):
        """
        Args:
            feature_maps (dict): Dictionary containing feature maps. 
                                 Keys should correspond to strides (e.g. 'scale_1', 'scale_16').
        Returns:
            all_norm_xy: [B, Total_Nodes, 2]
            all_reliability: [B, Total_Nodes] - The reliability weights used for sampling.
            all_features: [B, Total_Nodes, C]
            all_scale_weights: [B, Total_Nodes, num_sat_scales]
        """
        
        batch_size = None
        
        reliability_maps = {}
        proj_feature_maps = {}

        for stride in self.strides:
            # Resolve key name
            key = f'scale_{stride}' 
            feature_head = self.feature_head
            reliability_head = self.reliability_head
            
            if key not in feature_maps:
                raise KeyError(f"Feature map for stride {stride} not found.")

            featmap = feature_maps[key]
            
            if batch_size is None:
                batch_size = featmap.shape[0]

            B, C, H, W = featmap.shape
            
            reliability_map = reliability_head(featmap) # [B, 1, H, W]
            
            proj_featmap = feature_head(featmap) # [B, C, H, W]

            reliability_maps[key] = reliability_map.squeeze(1)  # [B, H, W]
            proj_feature_maps[key] = proj_featmap

        return reliability_maps, proj_feature_maps