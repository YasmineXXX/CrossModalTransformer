import torch.nn as nn
import torch


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        '''self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)'''

    def forward(self, input_ids):
        input_shape = input_ids.size()

        seq_length = input_shape[1]
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.position_embeddings(position_ids)
        # print(position_embeddings)

        inputs_embeds = self.word_embeddings(input_ids)
        # print(inputs_embeds)

        embeddings = (inputs_embeds + position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings.requires_grad_()

        return embeddings