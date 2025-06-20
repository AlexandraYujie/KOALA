import torch.nn as nn
# class SoftEmbedding(nn.Module):
#     def __init__(self,
#                  n_tokens,
#                  embedding_dim
#                  ):
#         super(SoftEmbedding, self).__init__()
#         self.embedding = nn.Parameter(n_tokens, embedding_dim)  #

def get_random_init_embedding(n_tokens, embedding_dim):
    random_init_embedding = nn.Parameter(n_tokens, embedding_dim)
    return random_init_embedding