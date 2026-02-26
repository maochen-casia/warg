import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)
sys.path.append(os.path.dirname(code_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from DINOv3.dinov3_encoder import DINOv3
from dpt import DPTHead
from graph_generator import GraphGenerator
from graph_matcher import GraphMatcher
from bev_transform import project_bev_graph, generate_candidate_graphs_by_translation

class WARG(nn.Module):
    """
        WARG: Warped Alignment for Reprojected Graphs
        
        Valar Morghulis. Valar Dohaeris.
    """

    def __init__(self, dino_model_name, num_nodes_per_scale, hid_dim, match_dim, 
                 train_search_steps, eval_search_steps, device):
        super().__init__()
        self.dino = DINOv3(model_name=dino_model_name, device=device, freeze=True)
        self.scale = self.dino.scale
        self.train_search_steps = train_search_steps
        self.eval_search_steps = eval_search_steps

        self.strides = [1, 2, 4, 8]
        self.dpt = DPTHead(in_channels=self.dino.embed_dim, features=hid_dim, final_out_channels=hid_dim,
                           out_channels=[hid_dim//2, hid_dim, hid_dim*2, hid_dim*2])

        self.graph_generator = GraphGenerator(num_nodes_per_scale=num_nodes_per_scale,
                                              hid_dim=hid_dim,
                                              out_dim=match_dim,
                                              num_sat_scales=len(self.strides),
                                              strides=self.strides)
        
        self.graph_matcher = GraphMatcher(strides=self.strides)

        self.device = device
        self.to(device)

    def train_params(self):

        params = [{'params': [p for p in self.parameters() if p.requires_grad], 'lr_scale': 1}]
        return params
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Overrides the default state_dict() method to exclude the parameters
        of the frozen 'dino' submodule.
        """
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        filtered_state_dict = OrderedDict()

        for key, value in original_state_dict.items():
            if not key.startswith(prefix + 'dino.'):
                filtered_state_dict[key] = value

        return filtered_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict() method to gracefully handle
        the missing 'dino' parameters by always using strict=False internally.
        """
        super().load_state_dict(state_dict, strict=False)

    def forward(self, data):

        left_image = data['left_image'].to(self.device)
        sat_image = data['sat_image'].to(self.device)
        left_depth_map = data['left_depth'].to(self.device)
        left_depth_map = data['left_depth'].to(self.device)
        K_left = data['K_left'].to(self.device)
        K_sat = data['K_sat'].to(self.device)
        R_left2world = data['R_left2world'].to(self.device)
        t_left2world_init = data['t_left2world_init'].to(self.device)
        R_sat2world = data['R_sat2world'].to(self.device)
        t_sat2world = data['t_sat2world'].to(self.device)
        max_init_offset = data['max_init_offset'].to(self.device)

        B, C, H1, W1 = left_image.shape
        B, C, H2, W2 = sat_image.shape

        left_intermediate_features = self.dino.get_intermediate_layers(left_image)
        left_multi_scale_featmaps = self.dpt(left_intermediate_features,
                                                    patch_size=(H1//self.scale, W1//self.scale),
                                                    out_size=(H1,W1))
        
        sat_intermediate_features = self.dino.get_intermediate_layers(sat_image)
        sat_multi_scale_featmaps = self.dpt(sat_intermediate_features,
                                                    patch_size=(H2//self.scale, W2//self.scale),
                                                    out_size=(H2,W2))

        node_norm_coords, node_reliability_weights, node_features = self.graph_generator.forward_left_featmaps(left_multi_scale_featmaps)
        sat_reliability_maps, sat_multi_scale_featmaps = self.graph_generator.forward_sat_featmaps(sat_multi_scale_featmaps)

        node_depths = F.grid_sample(left_depth_map.unsqueeze(1), 
                                   node_norm_coords.unsqueeze(1), 
                                   mode='bilinear', 
                                   align_corners=False).view(node_reliability_weights.shape)

        node_coords = (node_norm_coords + 1) / 2 * torch.tensor([W1 - 1, H1 - 1], device=self.device)

        if self.training:
            t_left2world_gt = data['t_left2world_gt'].to(self.device)
            candidate_node_gt_coords = project_bev_graph(
                                        node_coords,
                                        node_depths,
                                        K_left,
                                        R_left2world,
                                        t_left2world_gt,
                                        K_sat,
                                        R_sat2world,
                                        t_sat2world).unsqueeze(1) # (B, 1, N, 2)
            candidate_node_gt_norm_coords = (candidate_node_gt_coords / torch.tensor([W2 - 1, H2 - 1], device=self.device)) * 2 - 1
            # negative samples around gt
            search_steps = self.train_search_steps
            total_candidate_t_left2world = {}
            total_candidate_node_norm_coords = {}
            combined_candidate_t_left2world = [t_left2world_gt.unsqueeze(1)]
            combined_candidate_node_norm_coords = [candidate_node_gt_norm_coords]

            for stride in self.strides:
                search_radius = torch.ones([B], device=self.device) * 8 * stride
                candidate_t_left2world, candidate_node_coords = generate_candidate_graphs_by_translation(
                                                                node_coords=node_coords,
                                                                node_depths=node_depths,
                                                                K_left=K_left,
                                                                R_left2world=R_left2world,
                                                                t_left2world_init=t_left2world_gt,
                                                                K_sat=K_sat,
                                                                R_sat2world=R_sat2world,
                                                                t_sat2world=t_sat2world,
                                                                search_radius=search_radius,
                                                                search_steps=search_steps) # (B, S, N, 2)
                candidate_node_norm_coords = (candidate_node_coords / torch.tensor([W2 - 1, H2 - 1], device=self.device)) * 2 - 1
                combined_candidate_t_left2world.append(candidate_t_left2world)
                combined_candidate_node_norm_coords.append(candidate_node_norm_coords)
                # add gt
                candidate_t_left2world = torch.cat([t_left2world_gt.unsqueeze(1), candidate_t_left2world], dim=1)
                candidate_node_norm_coords = torch.cat([candidate_node_gt_norm_coords, candidate_node_norm_coords], dim=1)
        
                key = f'scale_{stride}'
                total_candidate_t_left2world[key] = candidate_t_left2world
                total_candidate_node_norm_coords[key] = candidate_node_norm_coords
            
            combined_candidate_t_left2world = torch.cat(combined_candidate_t_left2world, dim=1)
            combined_candidate_node_norm_coords = torch.cat(combined_candidate_node_norm_coords, dim=1)
            total_candidate_t_left2world['combined'] = combined_candidate_t_left2world
            total_candidate_node_norm_coords['combined'] = combined_candidate_node_norm_coords

            total_scores = self.graph_matcher.forward_train(
                                        node_reliability_weights=node_reliability_weights,
                                        node_features=node_features,
                                        sat_reliability_maps=sat_reliability_maps,
                                        sat_featmaps=sat_multi_scale_featmaps,
                                        total_candidate_node_norm_coords=total_candidate_node_norm_coords)
            
            t_left2world_pred = {}
            for key in total_scores.keys():
                scores = total_scores[key]
                candidate_t_left2world = total_candidate_t_left2world[key]
                t_left2world_pred[key] = candidate_t_left2world[torch.arange(B), torch.argmax(scores, dim=1)] # (B, 3)

            pred = {
                'total_scores_dict': total_scores,
                't_left2world_dict': t_left2world_pred
            }
            return pred
        
        else:
            search_radius = max_init_offset
            search_steps = self.eval_search_steps

            candidate_t_left2world, candidate_node_coords = generate_candidate_graphs_by_translation(
                                        node_coords=node_coords,
                                        node_depths=node_depths,
                                        K_left=K_left,
                                        R_left2world=R_left2world,
                                        t_left2world_init=t_left2world_init,
                                        K_sat=K_sat,
                                        R_sat2world=R_sat2world,
                                        t_sat2world=t_sat2world,
                                        search_radius=search_radius,
                                        search_steps=search_steps)
            
            candidate_node_norm_coords = (candidate_node_coords / torch.tensor([W2 - 1, H2 - 1], device=self.device)) * 2 - 1
            scores = self.graph_matcher.forward_inference(
                                        node_reliability_weights=node_reliability_weights,
                                        node_features=node_features,
                                        sat_reliability_maps=sat_reliability_maps,
                                        sat_featmaps=sat_multi_scale_featmaps,
                                        candidate_node_norm_coords=candidate_node_norm_coords
            )

            t_left2world = candidate_t_left2world[torch.arange(B), torch.argmax(scores, dim=1)] # (B, 3)
            pred = {'t_left2world': t_left2world}
            return pred

    def loss(self, pred, label):

        t_left2world_label = label['t_left2world'].to(self.device)

        t_left2world_dict = pred['t_left2world_dict']
        total_scores_dict = pred['total_scores_dict']
        total_nll_dict = {}
        total_error_dict = {}

        for key in total_scores_dict.keys():
            scores = total_scores_dict[key]
            nll = -torch.log(scores[:, 0] + 1e-8).mean()
            error = torch.norm(t_left2world_label - t_left2world_dict[key], dim=-1).mean()
            total_nll_dict[key] = nll
            total_error_dict[key] = error
        
        loss = 0

        for key, nll in total_nll_dict.items():
            loss = loss + nll
        
        # loss dict for display
        loss_dict = {'loss': loss.item(), 'nll_c': total_nll_dict['combined'].item(),
                     'err_c': total_error_dict['combined'].item()}

        return loss, loss_dict