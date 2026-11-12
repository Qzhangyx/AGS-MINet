
CUDA_LAUNCH_BLOCKING = 1
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_std
import numpy as np
from .radial_basis import RadialBasis
import copy
from math import pi as PI

from ..common import GaussianSmearing, ShiftedSoftplus
from ..protein_features import ProteinFeatures
from .tools1 import FrameAveraging, convert_to_batch_format, convert_from_batch_format, convert_h_to_residue_frame_format, convert_h_residue_back_to_batch
from .tools1 import convert_complex_to_batch_format, split_h_all, split_X_reconstructed_batch, split_X_batch

residue_atom_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]).float()

class TransitionBlock(nn.Module):
    def __init__(self, dim: int, input_dim: int = None):
        """
        Similar to AlphaFold3's Conditioned Transition Block
        """
        super().__init__()
        if input_dim is None:
            input_dim = dim
        self.ln = nn.LayerNorm(input_dim)
        self.swish_linear = nn.Linear(input_dim, dim, bias=False)
        self.swish_scale_linear = nn.Linear(input_dim, dim, bias=False)
        self.sigmoid_linear = nn.Linear(input_dim, dim, bias=True)
        self.sigmoid_scale_linear = nn.Linear(dim, dim, bias=False)
        nn.init.constant_(self.sigmoid_linear.bias, -2.0)

    # @torch.compile
    def forward(self, x):
        x = self.ln(x)
        b = F.silu(self.swish_linear(x)) * self.swish_scale_linear(x)
        x = F.sigmoid(self.sigmoid_linear(x)) * self.sigmoid_scale_linear(b)
        return x
    
class GETLayer(Module):
    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=8,
                 cutoff=10.0, device='cuda:0', sparse_k=3):
        super().__init__()

        self.consider_ligand = False
        self.sparse_k = sparse_k

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.d = int(self.hidden_channels / self.num_heads)
        self.k = k
        self.cutoff = cutoff
        self.device = device
        self.esm_refine = True

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)

        # Residue level settings
        self.atom_pos_embedding = nn.Embedding(14, self.hidden_channels)
        self.residue_embedding = nn.Embedding(21, self.hidden_channels)  # one embedding for mask
        self.W_Q = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_K = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_V = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_Q_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_K_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_V_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

        self.W_Q_f = nn.Linear(self.hidden_channels, 3*self.num_heads, bias=False)
        self.W_K_f = nn.Linear(self.hidden_channels, 3*self.num_heads, bias=False)
        self.W_V_f = nn.Linear(self.hidden_channels, 3*self.num_heads, bias=False)

        self.W_Q_f_lig = nn.Linear(self.hidden_channels, 3*self.num_heads, bias=False)
        self.W_K_f_lig = nn.Linear(self.hidden_channels, 3*self.num_heads, bias=False)
        self.W_V_f_lig = nn.Linear(self.hidden_channels, 3*self.num_heads, bias=False)

        self.W_O = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.W_O_lig = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.W_O_lig1 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.act = nn.SiLU()
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.layernorm1 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.1)

        self.T_i = nn.Sequential(nn.Linear(3 * self.d, self.d), nn.SiLU(),
                                 nn.Linear(self.d, 1))
        self.T_e1 = nn.Sequential(nn.Linear(3 * self.d, self.d), nn.SiLU(),
                                  nn.Linear(self.d, 1))
        self.T_e2 = nn.Sequential(nn.Linear(3 * self.hidden_channels, self.hidden_channels), nn.SiLU(),
                                  nn.Linear(self.hidden_channels, self.num_heads))
        self.residue_mlp = Linear(hidden_channels, 20, bias=True)
        self.sigma_D = Sequential(
            Linear(edge_channels, hidden_channels),
            nn.SiLU(),
            Linear(hidden_channels, self.num_heads),
        )
        self.sigma_D1 = Sequential(
            Linear(edge_channels, hidden_channels),
            nn.SiLU(),
            Linear(hidden_channels, self.num_heads),
        )
        self.sigma_v = Linear(edge_channels, hidden_channels)
        self.block_mlp_invariant = nn.Sequential(nn.Linear(self.d, self.d), nn.SiLU(), nn.Linear(self.d, self.d))
        self.block_mlp_equivariant = nn.Sequential(nn.Linear(self.d, self.d), nn.SiLU(), nn.Linear(self.d, self.d))
        self.att_feat_weight1 = nn.Parameter(torch.tensor(1.0))
        self.att_geo_weight1 = nn.Parameter(torch.tensor(0.1))  
        self.att_feat_weight2 = nn.Parameter(torch.tensor(1.0))
        self.att_geo_weight2 = nn.Parameter(torch.tensor(0.1))  
        self.TransitionBlock = TransitionBlock(hidden_channels, hidden_channels)

    @property
    def out_channels(self):
        return self.hidden_channels

    def connect_edges(self, res_X, batch):
        edge_index = knn_graph(res_X[:, 1], k=self.k, batch=batch, flow='target_to_source')
        edge_index, _ = add_self_loops(edge_index, num_nodes=res_X.size(0))  # add self loops
        return edge_index

    def extract_with_batch(self, X_ij, rows, depth, height, top_indices):
        """
        X_ij: [58, 4, 8, 14, 14] or [58, 4, 8, 14, 14, 3]
        """
        
        if X_ij.dim() == 6:
            
            results = []
            for b in range(X_ij.shape[2]):
                result_b = X_ij[:, :, b, :, :, :][rows, depth, height, top_indices, :]
                results.append(result_b)
            return torch.stack(results, dim=2)
        
        elif X_ij.dim() == 5:
           
            results = []
            for b in range(X_ij.shape[2]):
                result_b = X_ij[:, :, b, :, :][rows, depth, height, top_indices]
                results.append(result_b)
            return torch.stack(results, dim=2)  # [58, 4, 8, 14, k]
        
        else:
            raise ValueError(f"Unsupported dimension: {X_ij.dim()}")


    def attention_res_ligand(self, res_H, res_X, atom_mask, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num,
                         residue_mask):

        X_batch, mask_batch, res_per_batch, max_atoms = convert_to_batch_format(res_X, atom_mask, batch)
        framer = FrameAveraging()
        framer_h, framer_F_ops, framer_center = framer.create_frame(X_batch, mask_batch)
        framer_pro = convert_h_to_residue_frame_format(framer_h, res_per_batch, max_atoms, n_atoms_per_res=14).permute(0, 2, 1, 3)
        framer_lig = framer.apply_frame_to_ligand(ligand_pos, ligand_mask, framer_F_ops, framer_center)

        edge_index = self.connect_edges(res_X, batch)
        row, col = edge_index
        R_ij = torch.cdist(res_X[row], res_X[col], p=2)
        dist_rep = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], res_X.shape[1], -1)
        n_nodes = res_H.shape[0]
        n_edges = edge_index.shape[1]
        n_channels = res_X.shape[1]
        Q = self.W_Q(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        K = self.W_K(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        V = self.W_V(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        Q_lig = self.W_Q_lig(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        
        Q_lig_f = self.W_Q_f_lig(res_H).view([n_nodes, n_channels, self.num_heads, 3])
        Q_f = self.W_Q_f(res_H).view([n_nodes, n_channels, self.num_heads, 3])
        K_f = self.W_K_f(res_H).view([n_nodes, n_channels, self.num_heads, 3])
        V_f = self.W_V_f(res_H).view([n_nodes, n_channels, self.num_heads, 3])
        num_frame = framer_pro.shape[1]
        probe_pro1 = Q_f.transpose(1, 2).unsqueeze(2).expand(-1, -1, num_frame, -1, -1) + framer_pro.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
        
        diff = probe_pro1[row].unsqueeze(4)-framer_pro.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)[col].unsqueeze(3)
        att_ = torch.norm(diff, dim=-1)
        unit_direction = diff / (att_.unsqueeze(-1) + 1e-8)
        att = -att_.mean(2)


        attend_logits = torch.matmul(Q[row].transpose(1, 2), K[col].permute(0, 2, 3, 1)).view(
            [n_edges, self.num_heads, n_channels, n_channels])
        attend_logits /= np.sqrt(self.d)
       
        attend_logits = self.att_feat_weight1 * attend_logits + self.att_geo_weight1 * att
        attend_mask = (atom_mask[row].unsqueeze(-1) @ atom_mask[col].unsqueeze(-2)).unsqueeze(1).repeat(1,
                                                                                                        self.num_heads,
                                                                                                        1, 1)

        # sparse attention, only keep top k=3
        attend_logits[torch.logical_not(attend_mask)] = -1e5  # do not sellect from entries not attend
        _, top_indices = torch.topk(attend_logits, self.sparse_k, dim=-1, largest=True)
        sparse_mask = torch.zeros_like(attend_logits, dtype=torch.bool)
        rows = torch.arange(attend_logits.size(0)).view(-1, 1, 1, 1).expand(-1, attend_logits.size(1),
                                                                            attend_logits.size(2), self.sparse_k)
        depth = torch.arange(attend_logits.size(1)).view(1, -1, 1, 1).expand(attend_logits.size(0), -1,
                                                                             attend_logits.size(2), self.sparse_k)
        height = torch.arange(attend_logits.size(2)).view(1, 1, -1, 1).expand(attend_logits.size(0),
                                                                              attend_logits.size(1), -1, self.sparse_k)
        sparse_mask[rows, depth, height, top_indices] = True
        attend_logits = attend_logits * sparse_mask
        attend_logits = attend_logits * attend_mask

        # calculate beta
        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ atom_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / (attend_mask * sparse_mask).sum(-1).sum(-1)  # take avarage over non-zero entries
        beta = scatter_softmax(r_ij, row, dim=0)  # [nnodes, num_heads]

        attend_logits = torch.softmax(attend_logits, dim=-1)
        attend_logits = attend_logits * attend_mask * sparse_mask
        attend_logits = attend_logits / (
                attend_logits.norm(1, dim=-1).unsqueeze(-1) + 1e-7)  # normalize over every rows

        alpha_ij_Vj = attend_logits @ V[col].transpose(1, 2)  # [nedges, num_heads, 14, d]
        alpha_ij_Vj = self.block_mlp_invariant(alpha_ij_Vj)

        # invariant res_H update
        update_H = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj, row, dim=0, dim_size=n_nodes)
        update_H = update_H.transpose(1, 2).reshape(n_nodes, n_channels, -1)  # (nnodes, 14, heads*d)
        res_H = res_H + self.W_O(update_H)
        res_H = res_H * atom_mask.unsqueeze(-1)  # set empty entry zeros

        ###################################################################################
        # equivariant res_X update

        X_ij = res_X[row].unsqueeze(-2) - res_X[col].unsqueeze(1)  # [nedges, 14, 14, 3]
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)
        X_ij = X_ij.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
        X_ij = X_ij * attend_mask.unsqueeze(-1)

        
        dist_rep = self.sigma_v(dist_rep).view([n_edges, n_channels, n_channels, self.num_heads, self.d]).permute(0, 3,
                                                                                                                  1, 2,
                                                                                                                  4)
        input = torch.cat(
            [(Q[row].transpose(1, 2))[rows, depth, height], (K[col].transpose(1, 2))[rows, depth, top_indices],
             dist_rep[rows, depth, height, top_indices]], dim=-1)
        f = self.T_i(input).view(n_edges, self.num_heads, n_channels, self.sparse_k)
        f = f.unsqueeze(-1) * X_ij[rows, depth, height, top_indices].view(n_edges, self.num_heads, n_channels,
                                                                          self.sparse_k, 3)
        f = (f * attend_logits[rows, depth, height, top_indices].unsqueeze(-1)).sum(-2)  # [nedges, num_heads, 14, 3]
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(
            scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * f, row, dim=0, dim_size=n_nodes).sum(1)[residue_mask],
            min=-3.0, max=3.0)
        res_X = res_X * atom_mask.unsqueeze(-1)  # set empty entries zeros
  


        ###############################################################################################################



        batch_size = batch.max().item() + 1
        n_nodes = res_H.shape[0]
        n_channels = res_X.shape[1]
        lig_channel = ligand_feat.shape[1]
        row = torch.arange(n_nodes).to(self.device)[residue_mask]
        col = torch.repeat_interleave(torch.arange(batch_size).to(self.device), edit_residue_num)
        n_edges = row.shape[0]
        
        K_lig = self.W_K_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])
        V_lig = self.W_V_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])
        
        K_lig_f = self.W_K_f_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, 3])
        V_lig_f = self.W_V_f_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, 3])

        R_ij = torch.cdist(res_X[row], ligand_pos[col], p=2)
        attend_mask = (atom_mask[row].unsqueeze(-1) @ ligand_mask[col].unsqueeze(-2))
        R_ij = R_ij * attend_mask
        dist_rep1 = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], ligand_pos.shape[1], -1)

        probe_pro = Q_lig_f.permute(0, 2, 1, 3).unsqueeze(2).expand(-1, -1, 8, -1, -1) + framer_pro.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
        pro_expand = probe_pro[row] #[6, 8, 14, 1, 3]
        lig_expand = framer_lig[col].unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)  # [6, 8, 1, 5, 3]
        att_ = -torch.norm(pro_expand.unsqueeze(4) - lig_expand.unsqueeze(3), dim=-1)
        att_ = att_.mean(2) 

        attend_logits = torch.matmul(Q_lig[row].transpose(1, 2), K_lig[col].permute(0, 2, 3, 1)).view(
            [n_edges, self.num_heads, n_channels, lig_channel])
        attend_logits /= np.sqrt(self.d)
        attend_logits = self.att_feat_weight2 * attend_logits + self.att_geo_weight2 * att_
        attend_mask = attend_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # sparse attention, only keep top k=3
        attend_logits[torch.logical_not(attend_mask)] = -1e5  # do not sellect from entries not attend
        _, top_indices = torch.topk(attend_logits, self.sparse_k, dim=-1, largest=True)
        sparse_mask = torch.zeros_like(attend_logits, dtype=torch.bool)
        rows = torch.arange(attend_logits.size(0)).view(-1, 1, 1, 1).expand(-1, attend_logits.size(1),
                                                                            attend_logits.size(2), self.sparse_k)
        depth = torch.arange(attend_logits.size(1)).view(1, -1, 1, 1).expand(attend_logits.size(0), -1,
                                                                             attend_logits.size(2), self.sparse_k)
        height = torch.arange(attend_logits.size(2)).view(1, 1, -1, 1).expand(attend_logits.size(0),
                                                                              attend_logits.size(1), -1, self.sparse_k)
        sparse_mask[rows, depth, height, top_indices] = True
        attend_logits = attend_logits * attend_mask
        attend_logits = attend_logits * sparse_mask
        #return_attend = copy.deepcopy(attend_logits)
        return_attend = attend_logits.detach().clone()

        # calculate beta
        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        ligand_mask_head = ligand_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ ligand_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / (attend_mask * sparse_mask).sum(-1).sum(-1)  # take avarage over non-zero entries
        beta = scatter_softmax(r_ij, col, dim=0)

        attend_logits_lig = torch.softmax(attend_logits, dim=-2) * attend_mask
        attend_logits_lig = attend_logits_lig / (attend_logits_lig.norm(1, dim=-2).unsqueeze(-2) + 1e-7)

        attend_logits = attend_logits * sparse_mask
        attend_logits_res = torch.softmax(attend_logits, dim=-1)
        attend_logits_res = attend_logits_res * attend_mask * sparse_mask
        attend_logits_res = attend_logits_res / (attend_logits_res.norm(1, dim=-1).unsqueeze(-1) + 1e-7)
        alpha_ij_Vj = attend_logits_res @ V_lig[col].transpose(1, 2)

        # invariant feature update
        res_H[residue_mask] = res_H[residue_mask] + self.W_O_lig(
            alpha_ij_Vj.transpose(1, 2).reshape(residue_mask.sum(), n_channels, -1))
        res_H = res_H * atom_mask.unsqueeze(-1)  # set empty entries zeros
        alpha_ij_Vj_lig = attend_logits_lig.transpose(-1, -2) @ V[row].transpose(1, 2)
        update_lig_feat = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj_lig, col, dim=0,
                                      dim_size=batch_size)
        ligand_feat = ligand_feat + self.W_O_lig1(update_lig_feat.transpose(1, 2).reshape(batch_size, lig_channel, -1))
        ligand_feat = ligand_feat * ligand_mask.unsqueeze(-1)  # set empty entries zeros

        ###################################################################################
        # equivariant res_X update

        X_ij = res_X[row].unsqueeze(-2) - ligand_pos[col].unsqueeze(1)
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)
        X_ij = X_ij.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
        X_ij = X_ij * attend_mask.unsqueeze(-1)

        dist_rep1 = self.sigma_v(dist_rep1).view([n_edges, n_channels, lig_channel, self.num_heads, self.d]).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2,
                                                                                                                     4)
        input = torch.cat(
            [(Q_lig[row].transpose(1, 2))[rows, depth, height], (K_lig[col].transpose(1, 2))[rows, depth, top_indices],
             dist_rep1[rows, depth, height, top_indices]], dim=-1)
        f_res = self.T_e1(input).view(n_edges, self.num_heads, n_channels, self.sparse_k)
        f_res = f_res.unsqueeze(-1) * X_ij[rows, depth, height, top_indices].view(n_edges, self.num_heads, n_channels,
                                                                                  self.sparse_k, 3)
        f_res = (f_res * attend_logits[rows, depth, height, top_indices].unsqueeze(-1)).sum(-2).sum(
            1)  # [nedges, 14, 3]
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(f_res, min=-3.0, max=3.0)
        res_X = res_X * atom_mask.unsqueeze(-1)  # set empty entries zeros

        p_idx, q_idx = torch.cartesian_prod(torch.arange(n_channels), torch.arange(lig_channel)).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        dist_rep1 = dist_rep1.permute(0, 2, 3, 1, 4)

        f_lig = self.T_e2(torch.cat(
            [Q_lig[row][:, p_idx].view(-1, self.hidden_channels), K_lig[col][:, q_idx].view(-1, self.hidden_channels),
             dist_rep1.view(-1, self.hidden_channels)], dim=-1)).view(n_edges, n_channels, lig_channel, self.num_heads)
        # attend_mask = (atom_mask[row].unsqueeze(-1) @ ligand_mask[col].unsqueeze(-2)).bool()
        f_lig = f_lig.permute(0, 3, 1, 2) * attend_logits_lig
        f_lig = (f_lig.unsqueeze(-1) * X_ij).sum(2)  # [n_edges, num_heads, lig_channel, 3]
        ligand_pos = ligand_pos + scatter_sum((beta.unsqueeze(-1).unsqueeze(-1) * f_lig).sum(1), col, dim=0,
                                              dim_size=batch_size)
        ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)  # set empty entries zeros
        return res_H, res_X, ligand_pos, ligand_feat, return_attend

    def forward(self, res_H, res_X, atom_mask, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask):
        #res_H, res_X = self.attention(res_H, res_X, atom_mask, batch, residue_mask)
        res_H, res_X, ligand_pos, ligand_feat, return_attend = self.attention_res_ligand(res_H, res_X, atom_mask, batch, ligand_pos,
                                                                      ligand_feat, ligand_mask, edit_residue_num,
                                                                      residue_mask)

        pred_res_type = self.residue_mlp(res_H[residue_mask].sum(-2))
        return res_H, res_X, ligand_pos, ligand_feat, pred_res_type, return_attend


class EquivariantFFN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_channel=1, n_rbf=16, act_fn=nn.SiLU(),
                 residual=True, dropout=0.1, constant=1, z_requires_grad=True) -> None:
        super().__init__()
        self.constant = constant
        self.residual = residual
        self.n_rbf = n_rbf

        self.mlp_h = nn.Sequential(
            nn.Linear(d_in * 2 + n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Dropout(dropout)
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(d_in * 2 + n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
            nn.Dropout(dropout)
        )

        if not z_requires_grad:
            for param in self.mlp_z.parameters():
                param.requires_grad = False

        self.rbf = RadialBasis(n_rbf, 7.0)

    def forward(self, H, X, atom_mask, residue_mask):
        '''
        :param H: [N, d_in]
        :param Z: [N, n_channel, 3]
        :param block_id: [Nu]
        '''

        radial, (X_c, X_o) = self._radial(X, atom_mask)  # [N, n_hidden_channel], ([N, 1, 3], [N, n_channel, 3]
        # H_c = scatter_mean(H, block_id, dim=0)[block_id]  # [N, d_in]
        H_c = (H * atom_mask.unsqueeze(-1)).sum(-2) / atom_mask.sum(-1).unsqueeze(-1)
        H_c = H_c.unsqueeze(-2).repeat(1, 14, 1)
        inputs = torch.cat([H, H_c, radial], dim=-1)  # [N, 14, d_in + d_in + n_rbf]

        H_update = self.mlp_h(inputs)

        H = H + H_update if self.residual else H_update

        X_update = X_c.unsqueeze(-2) + self.mlp_z(inputs) * X_o
        X[residue_mask] = X_update[residue_mask]

        H, X = H * atom_mask.unsqueeze(-1), X * atom_mask.unsqueeze(-1)

        return H, X

    def _radial(self, X, atom_mask):
        X_c = (X * atom_mask.unsqueeze(-1)).sum(-2) / atom_mask.sum(-1).unsqueeze(-1)  # center
        X_o = X - X_c.unsqueeze(-2)  # [N, 14, 3], no translation
        X_o = X_o * atom_mask.unsqueeze(-1)

        D = stable_norm(X_o, dim=-1)  # [N, 14]
        radial = self.rbf(D.view(-1)).view(D.shape[0], D.shape[1], -1)  # [N, 14, n_rbf]
        return radial, (X_c, X_o)


class InvariantLayerNorm(nn.Module):
    def __init__(self, d_hidden) -> None:
        super().__init__()

        self.layernorm = nn.LayerNorm(d_hidden)
        self.layernorm1 = nn.LayerNorm(d_hidden)

    def forward(self, res_H, ligand_feat, atom_mask, ligand_mask):
        res_H[atom_mask.bool()] = self.layernorm(res_H[atom_mask.bool()])
        ligand_feat[ligand_mask.bool()] = self.layernorm1(ligand_feat[ligand_mask.bool()])
        return res_H, ligand_feat


class GET(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=8,
                 cutoff=10.0, device='cuda:0', n_layers=4, pre_norm=False, sparse_k=3):
        super().__init__()
        '''
        :param d_hidden: Number of hidden features
        :param d_radial: Number of features for calculating geometric relations
        :param n_channel: Number of channels of coordinates of each unit
        :param n_rbf: Dimension of RBF feature, 1 for not using rbf
        :param cutoff: cutoff for RBF
        :param d_edge: Number of features for the edge features
        :param n_layers: Number of layer
        :param act_fn: Non-linearity
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        '''

        self.n_layers = n_layers
        self.pre_norm = pre_norm
        self.sparse_k = sparse_k
        self.residue_atom_mask = residue_atom_mask.to(device)

        if self.pre_norm:
            self.pre_layernorm = InvariantLayerNorm(hidden_channels)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}',
                            GETLayer(hidden_channels, edge_channels, key_channels, num_heads, num_interactions, k,
                                     cutoff, device))
            self.add_module(f'layernorm0_{i}', InvariantLayerNorm(hidden_channels))
            self.add_module(f'ffn_{i}', EquivariantFFN(
                hidden_channels, 2 * hidden_channels, hidden_channels,
            ))
            self.add_module(f'layernorm1_{i}', InvariantLayerNorm(hidden_channels))

    def forward(self, res_H, res_X, res_S, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask):
        atom_mask = self.residue_atom_mask[res_S]

        #if self.pre_norm:
            #res_H, ligand_feat = self.pre_layernorm(res_H, ligand_feat, atom_mask, ligand_mask)

        for i in range(self.n_layers):
            res_H, res_X, ligand_pos, ligand_feat, pred_res_type, attend = self._modules[f'layer_{i}'](res_H, res_X, atom_mask,
                                                                                               batch, ligand_pos,
                                                                                               ligand_feat, ligand_mask,
                                                                                               edit_residue_num,
                                                                                               residue_mask)
            #res_H, ligand_feat = self._modules[f'layernorm0_{i}'](res_H, ligand_feat, atom_mask, ligand_mask)
            #res_H, res_X = self._modules[f'ffn_{i}'](res_H, res_X, atom_mask, residue_mask)
            #res_H, ligand_feat = self._modules[f'layernorm1_{i}'](res_H, ligand_feat, atom_mask, ligand_mask)

        return res_H, res_X, ligand_pos, ligand_feat, pred_res_type, attend


def stable_norm(input, *args, **kwargs):
    return torch.norm(input, *args, **kwargs)
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)


if __name__ == '__main__':
    hidden_channels = 128
    edge_channels = 64
    key_channels = 128
    num_heads = 4
    device = torch.device('cuda:0')
    model = GET()
    model.to(device)
    model.eval()

    res_H = torch.rand(10, 14, hidden_channels).to(device)
    res_X = torch.rand(10, 14, 3).to(device)
    res_S = torch.ones(10, dtype=torch.long)
    atom_mask = residue_atom_mask[res_S].to(device)
    ligand_mask = torch.tensor([[1., 1, 1, 0, 0], [1, 1, 1, 1, 1]]).to(device)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], device=device)
    residue_mask = torch.ones(10, device=device).bool()
    edit_residue_num = torch.tensor([3, 7], device=device)
    ligand_pos, ligand_feat = torch.rand(2, 5, 3).to(device), torch.rand(2, 5, hidden_channels).to(device)
    ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)
    res_X = res_X * atom_mask.unsqueeze(-1)

    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    res_H1, res_X1, ligand_pos1, ligand_feat1, pred_res_type1,_ = model(copy.deepcopy(res_H), res_X, res_S, batch,
                                                                      ligand_pos, copy.deepcopy(ligand_feat),
                                                                      ligand_mask, edit_residue_num,
                                                                      residue_mask)

    res_X1 = copy.deepcopy(res_X1.detach())
    res_X1[batch == 0] = torch.matmul(res_X1[batch == 0], Q1) + t1
    res_X1[batch == 1] = torch.matmul(res_X1[batch == 1], Q2) + t2
    res_X1 = res_X1 * atom_mask.unsqueeze(-1)

    ligand_pos1 = copy.deepcopy(ligand_pos1.detach())
    ligand_pos1[0] = torch.matmul(ligand_pos1[0], Q1) + t1
    ligand_pos1[1] = torch.matmul(ligand_pos1[1], Q2) + t2
    ligand_pos1 = ligand_pos1 * ligand_mask.unsqueeze(-1)

    res_X[batch == 0] = torch.matmul(res_X[batch == 0], Q1) + t1
    res_X[batch == 1] = torch.matmul(res_X[batch == 1], Q2) + t2
    res_X = res_X * atom_mask.unsqueeze(-1)

    ligand_pos[0] = torch.matmul(ligand_pos[0], Q1) + t1
    ligand_pos[1] = torch.matmul(ligand_pos[1], Q2) + t2
    ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)

    res_H2, res_X2, ligand_pos2, ligand_feat2, pred_res_type2,_ = model(copy.deepcopy(res_H), res_X, res_S, batch,
                                                                      ligand_pos, copy.deepcopy(ligand_feat),
                                                                      ligand_mask, edit_residue_num,
                                                                      residue_mask)

    print((res_X1 - res_X2).norm())
    print((res_H1 - res_H2).float().norm())
    print((pred_res_type1 - pred_res_type2).norm())
    print((ligand_pos1 - ligand_pos2).norm())
    print((ligand_feat1 - ligand_feat2).norm())


