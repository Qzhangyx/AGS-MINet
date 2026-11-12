import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_std
class FrameAveraging(nn.Module):

    def __init__(self):
        super().__init__()
        self.ops = torch.tensor(
            [[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
        )

    def create_frame(self, X, mask):
        device = X.device
        mask = mask.unsqueeze(-1).float()
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B,3,3] (Cov)
        _, V = torch.linalg.eigh(C.detach(), UPLO="U")  # [B,3,3]
        self.ops = self.ops.to(device)
        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(
            1
        )  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,8,N,3]
        
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)
    

    def apply_frame_to_ligand(self, ligand_pos, ligand_mask, F_ops, center):

        ligand_mask = ligand_mask.unsqueeze(-1).float()
        ligand_centered = ligand_pos - center.unsqueeze(1) * ligand_mask
        h_lig = torch.einsum("boij,bmj->bomi", F_ops.transpose(2, 3), ligand_centered)
        
        return h_lig
    

def convert_to_batch_format(X, mask, batch_id):
    max_residue_num = X.shape[1]
    num_batch = batch_id.max().item() + 1
    res_per_batch = torch.bincount(batch_id)  # [B]
    max_res = res_per_batch.max().item()
    max_atoms = max_res * max_residue_num
    
    X_batch = torch.zeros(num_batch, max_atoms, 3, device=X.device, dtype=X.dtype)
    mask_batch = torch.zeros(num_batch, max_atoms, dtype=torch.bool, device=X.device)
    
    for b in range(num_batch):
        res_mask = (batch_id == b)
        n_res = res_mask.sum().item()
        if n_res == 0:
            continue
        X_batch[b, :n_res*max_residue_num] = X[res_mask].reshape(-1, 3)
        mask_batch[b, :n_res*max_residue_num] = mask[res_mask].reshape(-1)
    
    return X_batch, mask_batch, res_per_batch, max_atoms


def convert_from_batch_format(X_batch, mask_batch, res_per_batch):

    X_list = []
    mask_list = []
    
    for b in range(len(res_per_batch)):
        n_res = res_per_batch[b].item()
        if n_res == 0:
            continue
        
        # 
        X_b = X_batch[b, :n_res*14].reshape(n_res, 14, 3)
        mask_b = mask_batch[b, :n_res*14].reshape(n_res, 14)
        
        X_list.append(X_b)
        mask_list.append(mask_b)
    
    X = torch.cat(X_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    
    return X, mask









def convert_complex_to_batch_format(protein_X, protein_mask, ligand_pos, ligand_mask, batch_id):

    num_batch = batch_id.max().item() + 1
    res_per_batch = torch.bincount(batch_id)  # [B]
    max_res = res_per_batch.max().item()
    max_res_atoms = max_res * 14
    max_lig_atoms = ligand_pos.shape[1]
    max_atoms = max_res_atoms + max_lig_atoms
    
    device = protein_X.device
    
    X_batch = torch.zeros(num_batch, max_atoms, 3, device=device, dtype=protein_X.dtype)
    mask_batch = torch.zeros(num_batch, max_atoms, dtype=torch.bool, device=device)
    
    protein_indices = []
    ligand_indices = []
    
    for b in range(num_batch):
        #  max_res_atoms
        protein_start = 0
        protein_end = max_res_atoms  # 
        
        res_mask = (batch_id == b)
        n_res = res_mask.sum().item()
        n_protein_atoms = n_res * 14
        
        if n_res > 0:
            # 
            X_batch[b, :n_protein_atoms] = protein_X[res_mask].reshape(-1, 3)
            mask_batch[b, :n_protein_atoms] = protein_mask[res_mask].reshape(-1)
        
        # 
        lig_start = max_res_atoms
        lig_end = lig_start + max_lig_atoms
        
        X_batch[b, lig_start:lig_end] = ligand_pos[b]
        mask_batch[b, lig_start:lig_end] = ligand_mask[b]
        
        protein_indices.append((protein_start, protein_end))
        ligand_indices.append((lig_start, lig_end))
    
    return X_batch, mask_batch, res_per_batch, max_atoms, protein_indices, ligand_indices

def split_h_all(h_all, max_res_atoms, max_lig_atoms):
    protein_h = h_all[:, :, :max_res_atoms, :]  # [B, 8, max_res_atoms, 3]
    ligand_h = h_all[:, :, max_res_atoms:, :]   # [B, 8, max_lig_atoms, 3]
    return protein_h, ligand_h

def split_X_reconstructed_batch(X_reconstructed_batch, max_res_atoms, max_lig_atoms):
    protein_h = X_reconstructed_batch[:, :max_res_atoms, :]  # [B, 8, max_res_atoms, 3]
    ligand_h = X_reconstructed_batch[:, max_res_atoms:, :]   # [B, 8, max_lig_atoms, 3]
    return protein_h, ligand_h

def split_X_batch(mask_batch, max_res_atoms, max_lig_atoms):
    protein_mask_batch = mask_batch[:, :max_res_atoms]  # [B, 8, max_res_atoms, 3]
    ligand_mask_batch = mask_batch[:, max_res_atoms:]   # [B, 8, max_lig_atoms, 3]
    return protein_mask_batch, ligand_mask_batch  




def convert_h_to_residue_frame_format(h_batch, res_per_batch,  max_atoms, n_atoms_per_res=14):
    """
     [B, frames, max_atoms, 3] -- [total_residues, n_atoms_per_res, frames, 3]
    
    Args:
        h_batch: [B, frames, max_atoms, 3] - create_frame
        res_per_batch: [B] -  batch 
        n_atoms_per_res: int - 14
    
    Returns:
        h_residue: [total_residues, n_atoms_per_res, frames, 3]
    """
    B, frames, max_atoms, _ = h_batch.shape
    total_residues = res_per_batch.sum().item()
    
    # 
    h_residue = torch.zeros(total_residues, n_atoms_per_res, frames, 3, 
                            device=h_batch.device, dtype=h_batch.dtype)
    
    residue_idx = 0
    for b in range(B):
        n_res = res_per_batch[b].item()
        n_atoms = n_res * n_atoms_per_res
        
        #  batch remove padding
        h_valid = h_batch[b, :, :n_atoms, :]  # [frames, n_atoms, 3]
        
        # reshape [n_res, n_atoms_per_res, frames, 3]
        h_reshaped = h_valid.permute(1, 0, 2).reshape(n_res, n_atoms_per_res, frames, 3)
        # h_valid [frames, n_atoms, 3] -> permute(1,0,2) [n_atoms, frames, 3] 
        #       -> reshape [n_res, n_atoms_per_res, frames, 3]
        
        # 
        h_residue[residue_idx:residue_idx + n_res] = h_reshaped
        residue_idx += n_res
    
    return h_residue


def convert_h_residue_back_to_batch(h_residue, res_per_batch, max_atoms, n_atoms_per_res=14):
    """
     [total_residues, n_atoms_per_res, frames, 3] -- [B, frames, max_atoms, 3]
    
    Args:
        h_residue: [total_residues, n_atoms_per_res, frames, 3]
        res_per_batch: [B] - batch residue_num
        max_atoms: int - each batch max atom numbers（using padding）
        n_atoms_per_res: int -  14
    
    Returns:
        h_batch: [B, frames, max_atoms, 3]
    """
    B = len(res_per_batch)
    frames = h_residue.shape[2]
    device = h_residue.device
    dtype = h_residue.dtype
    
    # 
    h_batch = torch.zeros(B, frames, max_atoms, 3, device=device, dtype=dtype)
    
    residue_idx = 0
    for b in range(B):
        n_res = res_per_batch[b].item()
        n_atoms = n_res * n_atoms_per_res
        
        # 
        h_reshaped = h_residue[residue_idx:residue_idx + n_res]  # [n_res, n_atoms_per_res, frames, 3]
        
        # reshape [n_atoms, frames, 3]
        h_valid = h_reshaped.reshape(n_atoms, frames, 3)  # [n_atoms, frames, 3]
        
        # T [frames, n_atoms, 3]
        h_valid = h_valid.permute(1, 0, 2)  # [frames, n_atoms, 3]
        
        # 
        h_batch[b, :, :n_atoms, :] = h_valid
        # 
        
        residue_idx += n_res
    
    return h_batch