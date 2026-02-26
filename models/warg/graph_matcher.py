import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphMatcher(nn.Module):

    def __init__(self, strides):
        super().__init__()

        self.strides = strides

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_train(self,
                      node_reliability_weights: torch.Tensor,
                      node_features: torch.Tensor,
                      sat_reliability_maps: dict[str, torch.Tensor],
                      sat_featmaps: dict[str, torch.Tensor],
                      total_candidate_node_norm_coords: dict[str, torch.Tensor]):
        
        B, N, _ = node_features.shape
        norm_node_features = F.normalize(node_features, p=2, dim=-1)  # (B, N, D)

        total_scores = {}

        for stride in self.strides:
            key = f'scale_{stride}'
            grid = total_candidate_node_norm_coords[key] # (B, K, N, 2)
            K = grid.shape[1]
            grid = grid.permute(0, 2, 1, 3).reshape(B, N * K, 1, 2)

            sat_reliability_map = sat_reliability_maps[key]  # (B, H, W)
            sampled_reliability_weights = F.grid_sample(
                sat_reliability_map.unsqueeze(1), 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            ).view(B, N, K)  # (B, N, K)

            sat_featmap = sat_featmaps[key]  # (B, C, H, W)
            sampled_feature = F.grid_sample(
                sat_featmap, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            ).view(B, sat_featmap.shape[1], N, K).permute(0, 2, 3, 1)  # (B, N, K, C)

            norm_sat_features = F.normalize(sampled_feature, p=2, dim=-1)  # (B, N, K, C)

            sim = torch.einsum('bnc, bnkc -> bnk', norm_node_features, norm_sat_features)  # (B, N, K)
            pair_reliability_weights = node_reliability_weights.unsqueeze(-1) + sampled_reliability_weights  # (B, N, K)
            
            pair_reliability_scores = F.softmax(pair_reliability_weights, dim=1)  # (B, N, K)

            logits = sim * pair_reliability_scores  # (B, N, K)
            logits = torch.sum(logits, dim=1) * self.logit_scale.exp()  # (B, K)
            scores = F.softmax(logits, dim=1)  # (B, K)

            total_scores[key] = scores

        combined_norm_node_coords = total_candidate_node_norm_coords['combined']
        combined_scores = self.forward_inference(
            node_reliability_weights,
            node_features,
            sat_reliability_maps,
            sat_featmaps,
            combined_norm_node_coords
        )

        total_scores['combined'] = combined_scores

        return total_scores

    def forward_inference(self,
                node_reliability_weights: torch.Tensor,
                node_features: torch.Tensor,
                sat_reliability_maps: dict[str, torch.Tensor],
                sat_featmaps: dict[str, torch.Tensor],
                candidate_node_norm_coords: torch.Tensor):

        B, N, _ = node_features.shape
        _, K, _, _ = candidate_node_norm_coords.shape

        norm_node_features = F.normalize(node_features, p=2, dim=-1)  # (B, N, D)
        
        grid = candidate_node_norm_coords.permute(0, 2, 1, 3).reshape(B, N * K, 1, 2)

        total_sim = []
        total_reliability = []

        for stride in self.strides:
            key = f'scale_{stride}'

            sat_reliability_map = sat_reliability_maps[key]  # (B, H, W)
            sampled_reliability_weights = F.grid_sample(
                sat_reliability_map.unsqueeze(1), 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            ).view(B, N, K)  # (B, N, K)

            sat_featmap = sat_featmaps[key]  # (B, C, H, W)
            sampled_feature = F.grid_sample(
                sat_featmap, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            ).view(B, sat_featmap.shape[1], N, K).permute(0, 2, 3, 1)  # (B, N, K, C)

            norm_sat_features = F.normalize(sampled_feature, p=2, dim=-1)  # (B, N, K, C)

            sim = torch.einsum('bnc, bnkc -> bnk', norm_node_features, norm_sat_features)  # (B, N, K)
            pair_reliability_weights = node_reliability_weights.unsqueeze(-1) + sampled_reliability_weights  # (B, N, K)
            
            total_sim.append(sim) # (B, N, K)
            total_reliability.append(pair_reliability_weights)  # (B, N, K)
        
        total_sim = torch.stack(total_sim, dim=-1)  # (B, N, K, S)
        total_reliability = torch.stack(total_reliability, dim=-1)  # (B, N, K, S)
        total_reliability = total_reliability.permute(0,2,1,3).reshape(B, K, N*len(self.strides))  # (B, K, N*S)
        total_reliability = F.softmax(total_reliability, dim=-1)  # (B, K, N*S)
        total_sim = total_sim.permute(0,2,1,3).reshape(B, K, N*len(self.strides))  # (B, K, N*S)
        candidate_logits = torch.sum(total_sim * total_reliability, dim=-1)  # (B, K)
        candidate_logits = candidate_logits * self.logit_scale.exp()
        candidate_scores = F.softmax(candidate_logits, dim=1)

        return candidate_scores