import torch.nn as nn
import torch

device = torch.device("cuda:0")

class VisualInputEmbedding(nn.Module):
    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config
        # sequence embedding
        self.row_position_embeddings = nn.Embedding(
            config.max_grid_row_position_embeddings,
            config.hidden_size)
        self.col_position_embeddings = nn.Embedding(
            config.max_grid_col_position_embeddings,
            config.hidden_size)
        self.LayerNorm = nn.BatchNorm2d(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, batch_video):
        """
        Args:
            batch_video: (B, C, H, W), process frame by frame of bsz videos

        Returns:

        """
        bsz, hsz, _, _ = batch_video.shape

        batch_video_pos_embed = self.add_2d_positional_embeddings(batch_video)

        batch_video_pos_embed = self.LayerNorm(batch_video_pos_embed)

        return batch_video_pos_embed

    def add_2d_positional_embeddings(self, batch_video):
        """
        Args:
            batch_video: (B, C, H, W)

        Returns:
            (B, d, H, W)
        """
        hsz, height, width = batch_video.shape[-3:]

        device = batch_video.device

        # add row-wise position embeddings
        row_position_ids = torch.arange(height, dtype=torch.long).to(device)  # (H, )
        #row_position_ids = torch.arange(height, dtype=torch.long)
        row_position_embeddings = self.row_position_embeddings(
            row_position_ids)  # (H, d)
        row_shape = (1, hsz, height, 1)
        # (1, d, H, 1)
        batch_video = batch_video + row_position_embeddings.view(*row_shape)  # broadcast automatically

        # add column-wise position embeddings
        col_position_ids = torch.arange(width, dtype=torch.long).to(device)  # (W, )
        #col_position_ids = torch.arange(width, dtype=torch.long)
        col_position_embeddings = self.col_position_embeddings(
            col_position_ids)  # (W, d)
        col_shape = (1, hsz, 1, width)  # (1, d, 1, W)
        batch_video = batch_video + col_position_embeddings.view(*col_shape)  # broadcast automatically
        return batch_video