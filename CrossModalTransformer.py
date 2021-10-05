import torch.nn as nn
from ResNet import ResNet, Bottleneck
from BertEmbeddings import BertEmbeddings
from VisualInputEmbedding import VisualInputEmbedding
from MultiheadAttention import MultiHeadedAttention

class CrossModalTransformer(nn.Module):
    def __init__(self, config):
        super(CrossModalTransformer, self).__init__()
        self.config = config
        self.frame_encoder = ResNet(block = Bottleneck,
                   layers = [3, 4, 6, 3],
                   grayscale = False)
        self.frame_embed = VisualInputEmbedding(config)
        self.query_embed = BertEmbeddings(config)
        self.multihead_att = MultiHeadedAttention(self.config.head, self.config.hidden_size, self.config.grid_size, self.config)

    def forward(self, frame, text):
        frame_enc = self.frame_encoder(frame)
        #frame_enc: bsz, hidden_size, 3, 3
        frame_vec = self.frame_embed(frame_enc)
        query_vec = self.query_embed(text)
        #query_vec: bsz, words, hidden_size

        query = query_vec
        #print(frame_vec.shape)
        key = frame_vec.view(-1, self.config.hidden_size, self.config.grid_size)
        value = frame_vec.view(-1, self.config.hidden_size, self.config.grid_size)
        #multihead_att = MultiHeadedAttention(self.config.head, self.config.hidden_size, k_w, self.config)
        #multihead_att.to(device)
        #multihead_att = torch.nn.DataParallel(multihead_att, device_ids=config.DEVICE_IDS)  # 声明所有可用设备
        #multihead_att = multihead_att.cuda(device=config.DEVICE_IDS[0])  # 模型放在主设备
        score = self.multihead_att(query, key, value)

        return score