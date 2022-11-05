
import heapq
import torch

class kg_context_infer_network(torch.nn.Module):
    def __init__(self,kg_emb_dim,d_inner,relation_num):
        super(kg_context_infer_network,self).__init__()
        self._norm_layer1 = torch.nn.Linear(kg_emb_dim * 7, d_inner * 2)  # 128*7(896)->2048
        self._relu = torch.nn.ReLU(inplace=True)
        self._norm_layer2 = torch.nn.Linear(d_inner * 2, int(d_inner/2))  # 2048->512
        self._norm_layer3 = torch.nn.Linear(int(d_inner/2), kg_emb_dim)  # 512->128
        self._norm_layer4 = torch.nn.Linear(kg_emb_dim, relation_num)  # 128->2

    def forward(self, es_rep,ed_rep,context_rep):
        o1 = torch.cat([es_rep, ed_rep, context_rep], dim=-1)
        o2 = torch.cat([torch.mul(es_rep, ed_rep), torch.mul(es_rep, context_rep),torch.mul(ed_rep, context_rep)], dim=-1)
        o3 = torch.mul(torch.mul(es_rep, ed_rep), context_rep)
        m = torch.cat([o1, o2, o3], dim=-1)

        pair_matrix = self._norm_layer1(m)
        pair_matrix = self._relu(pair_matrix)
        pair_matrix = self._norm_layer2(pair_matrix)
        pair_matrix = self._relu(pair_matrix)
        pair_matrix = self._norm_layer3(pair_matrix)
        pair_matrix = self._relu(pair_matrix)
        relation_matrix = self._norm_layer4(pair_matrix)
        return relation_matrix

def expand_if_not_none(tensor, dim, beam_width):
    """expand tensor dimension
    Args:
        tensor: torch.Tensor
        dim: int
        beam_width: int
    """
    if tensor is None:
        return None
    tensor_shape = list(tensor.shape)
    tensor = tensor.unsqueeze(dim + 1)
    expand_dims = [-1] * (len(tensor_shape) + 1)
    expand_dims[dim + 1] = beam_width
    tensor = tensor.expand(*expand_dims)
    tensor_shape[dim] = tensor_shape[dim] * beam_width
    tensor = tensor.reshape(*tensor_shape)
    return tensor.contiguous()


def repeat_if_not_none(tensor, dim, beam_width):
    """repeat tensor dimension
    Args:
        tensor: torch.Tensor
        dim: int
        beam_width: int
    """
    if tensor is None:
        return None
    tensor_shape = list(tensor.shape)
    tensor = tensor.unsqueeze(dim + 1)
    expand_dims = [1] * (len(tensor_shape) + 1)
    expand_dims[dim + 1] = beam_width
    tensor = tensor.repeat(*expand_dims)
    tensor_shape[dim] = tensor_shape[dim] * beam_width
    tensor = tensor.reshape(*tensor_shape)
    return tensor


class Branch:
    def __init__(self, score, tensor, length, alpha=1.0, log_act=True):
        self.score = Branch.normal_score(score, length, alpha, log_act)
        self.tensor = tensor

    def __lt__(self, other):
        return self.score <= other.score

    def __eq__(self, other):
        return self.score == other.score

    def __gt__(self, other):
        return self.score >= other.score

    @staticmethod
    def normal_score(score, length, alpha=1.0, log_act=True):
        assert alpha >= 0.0, "alpha should >= 0.0"
        assert alpha <= 1.0, "alpha should <= 1.0"

        if log_act:
            score = score / (length ** alpha)
        else:
            score = score ** (1 / (length ** alpha))

        return score

    def get_tensor(self):
        return self.tensor


class MatureBucket:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.bucket = []

    def push(self, item: Branch):
        if len(self.bucket) < self.bucket_size:
            heapq.heappush(self.bucket, item)
        else:
            if item.score > self.bucket[0].score:
                heapq.heappushpop(self.bucket, item)

    def get_max(self):
        self.bucket = sorted(self.bucket, reverse=True)
        return self.bucket[0].get_tensor()

def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)