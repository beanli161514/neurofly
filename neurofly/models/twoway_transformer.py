import torch
import math
import numpy as np
from scipy.ndimage import generic_filter
from skimage.morphology import ball
from torch import nn
from typing import Tuple, Optional

class TwoWayTransformer(nn.Module):
    def __init__(self, 
                 traj_input_dim: int, 
                 img_input_channels: int, 
                 embedding_dim: int, 
                 num_heads: int, 
                 mlp_dim: int, 
                 depth: int,
                 patch_size: int,
                 ):
        super(TwoWayTransformer, self).__init__()
        
        # Learned query token
        self.query_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        # Positional Embeddings
        self.position_embed = PositionEmbeddingRandom3D(num_pos_feats=embedding_dim // 2)
        self.order_embed = nn.Embedding(5, embedding_dim)  # assuming max 4 points

        # Embedding layers for trajectory and image patches
        self.traj_embed = nn.Linear(traj_input_dim, embedding_dim)
        self.img_embed = nn.Conv3d(img_input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=patch_size//2)

        # Transformer layers
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        self.attn_layers = nn.ModuleList([TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim) for _ in range(depth)])

        # Final MLP to output future trajectory points
        self.fc = nn.Linear(embedding_dim, 3)

    def forward(self, traj_seq, img):
        b,c,d,w,h = img.shape
        traj_pos_embed = self.position_embed.forward_with_coords(traj_seq, (d,h,w))
        # batch_size, n_points, dim
        traj_order = torch.arange(traj_seq.size(1), device=traj_seq.device)
        traj_order_embed = self.order_embed(traj_order)
        traj_embeds = traj_pos_embed + traj_order_embed

        img_embeds = self.img_embed(img)  # (batch_size, embedding_dim, D', H', W')
        d_out, h_out, w_out = img_embeds.shape[-3:]
        img_pos_embed = self.position_embed((d_out, h_out, w_out))  # (embedding_dim, D', H', W')
        img_embeds = img_embeds.flatten(2).permute(0, 2, 1)  # (batch_size, num_patches, embedding_dim)
        img_embeds = img_embeds + img_pos_embed.view(-1, self.embedding_dim)

        # Expand query token for batch size
        query_token = self.query_token.expand(b, -1, -1)  # (batch_size, 1, embedding_dim)

        # Apply Transformer Blocks
        for layer in self.attn_layers:
            traj_embeds, img_embeds, query_token = layer(traj_embeds, img_embeds, query_token)

        # Use query token to predict the next trajectory point
        future_waypoints = self.fc(query_token.squeeze(1))
        return future_waypoints



class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim):
        super(TwoWayAttentionBlock, self).__init__()
        
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads)
        
        # Cross-attention for query token
        self.cross_attn_query = Attention(embedding_dim, num_heads)
        self.norm5 = nn.LayerNorm(embedding_dim)

    def forward(self, queries, keys, query_token):
        # Self-attention for trajectory points
        queries = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries)

        # Cross-attention (trajectory points to image)
        attn_out = self.cross_attn_token_to_image(queries, keys, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross-attention (image to trajectory points)
        attn_out = self.cross_attn_image_to_token(keys, queries, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        # Query token attends to both trajectory and image features
        query_token = self.cross_attn_query(query_token, torch.cat([queries, keys], dim=1), torch.cat([queries, keys], dim=1))
        query_token = self.norm5(query_token)

        return queries, keys, query_token

class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)



class PositionEmbeddingRandom3D(nn.Module):
    """
    3D Positional encoding using random spatial frequencies, similar to SAM's 2D version.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),  # 3D version (Z, Y, X)
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1  # Map to [-1,1]
        coords = coords @ self.positional_encoding_gaussian_matrix  # (D, H, W, C)
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # (D, H, W, 2C)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified 3D size."""
        d, h, w = size
        device = self.positional_encoding_gaussian_matrix.device

        # Create normalized coordinate grid
        grid = torch.ones((d, h, w), device=device, dtype=torch.float32)
        z_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        x_embed = grid.cumsum(dim=2) - 0.5
        z_embed /= d
        y_embed /= h
        x_embed /= w

        # Stack into shape (D, H, W, 3)
        pe = self._pe_encoding(torch.stack([z_embed, y_embed, x_embed], dim=-1))

        return pe

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        d, h, w = image_size
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / w
        coords[:, :, 1] = coords[:, :, 1] / h 
        coords[:, :, 2] = coords[:, :, 2] / d 
        return self._pe_encoding(coords.to(torch.float))  # B x N x C



class PosPredictor():
    def __init__(self,ckpt_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else self.device

        model = TwoWayTransformer(
            traj_input_dim=3, 
            img_input_channels=2, 
            embedding_dim=128, 
            num_heads=4, 
            mlp_dim=256, 
            depth=16,
            patch_size=4
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model

    
    def predict_displacement(self,traj,img):
        # input img: ndarray [0,65535]
        # input traj: list [5,3], in global coordinates
        # output displacement: list [dx,dy,dz]
        r = 16
        traj = [[i - j + r for i, j in zip(traj_point, traj[-1])] for traj_point in traj]
        traj = np.array(traj)
        img = preprocess(img)
        traj_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, context_length, 3)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 1, D, H, W)
        displacement = list(self.model(traj_tensor, img_tensor).detach().cpu().numpy()[0])
        return displacement


def rank_of_center(voxel_values):
    center_value = voxel_values[len(voxel_values) // 2]
    rank = np.sum(voxel_values < center_value)
    return rank


def preprocess(img):
    footprint = ball(5)
    filtered_image = generic_filter(img, rank_of_center, footprint=footprint, mode='mirror')
    filtered_image = filtered_image / np.sum(footprint)
    min_val = img.min()
    max_val = img.max()
    min_max = (img - min_val) / (max_val - min_val)
    img_combined = np.stack([filtered_image, min_max], axis=0)
    return img_combined
