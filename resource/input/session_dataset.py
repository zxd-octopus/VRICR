import numpy as np
from torch.utils.data.dataset import Dataset
from resource.option.dataset_option import DatasetOption as DO
from resource.option.train_option import TrainOption as TO
from copy import deepcopy
import json
import logging
import torch
from torch.utils.data._utils.collate import default_collate
from itertools import permutations
s_logger = logging.getLogger("main.dataset")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
def clip_pad_sentence(sentence,
                      max_len,
                      sos=DO.PreventWord.SOS_ID,
                      eos=DO.PreventWord.EOS_ID,
                      pad=DO.PreventWord.PAD_ID,
                      save_prefix=False):

    if sos is not None and eos is not None:
        max_len_ = max_len - 2  # cls
    elif sos is None and eos is None:
        max_len_ = max_len
    else:
        max_len_ = max_len-1

    if save_prefix:
        sentence = sentence[:max_len_]
    else:
        sentence = sentence[-max_len_:]

    if sos is not None:
        sentence = [sos] + sentence
    if eos is not None:
        sentence = sentence + [eos]

    sentence_length = len(sentence)
    sentence = sentence + [pad] * (max_len - sentence_length)
    sentence_mask = [1] * sentence_length + [0] * (max_len - sentence_length)

    return sentence, sentence_length,sentence_mask

def my_collate_fn(batch):
    all_movies = [x[0] for x in batch]
    identities = [x[1] for x in batch]
    batch = [ x[2:] for x in batch]

    batch = default_collate(batch)
    return [batch,all_movies,identities]


class SessionDataset(Dataset):
    def __init__(self,
                 sessions,
                 topic_to_id,
                 tokenizer,
                 toca,
                 mode="train",
                 relation_num = None,
                 graph = None,
                 entity2neighbor = None,
                 topic_num = None
                 ):
        super(SessionDataset, self).__init__()
        self.sessions = sessions
        self.mode = mode
        self.tokenizer=tokenizer
        self.is_train = (mode == "train")
        self.topic_to_id=topic_to_id
        self.topic_class_num = len(self.topic_to_id) if topic_num is None else topic_num
        self.id_to_topic = {
            id: topic
            for topic, id in self.topic_to_id.items()
        }
        self.toca=toca
        self.not_hit=0
        self.graph = graph
        self.relation_num = relation_num
        self.entity2neighbor = entity2neighbor


    def get_topic_graph(self):
        topic_graph = json.load(open(DO.graph_two_hop))
        return topic_graph

    def get_related_movies(self, topic_path, movie_num):
        neighbors = []
        for topic in topic_path:
            if topic in self.entity2neighbor:
                neighbor = self.entity2neighbor[topic]
                neighbors.extend(neighbor)
        neighbors = list(set(neighbors))
        related_topic = neighbors[-movie_num:]
        return related_topic

    def epair_to_target(self,ed_index,es_index,es_len):

        es_num = len(es_index)
        ed_num = len(ed_index)

        targets = torch.zeros(ed_num, es_num, 2)
        targets[:,:,0] = 1

        for es_tmp in range(es_len):
            es = es_index[es_tmp]
            target_ = self.graph[es]
            for ed_ex in target_:
                try:
                    ed_tmp = ed_index.index(ed_ex)
                    targets[ed_tmp,es_tmp,:] = torch.tensor([0, 1])
                except:
                    pass
        return targets

    def __getitem__(self, index):

        case = deepcopy(self.sessions[index])
        identity  = case["identity"]
        context = case["context_tokens"]
        topic_path = case["context_entities"]
        target = case["items"]
        response = case["response"]
        context_gen = case["context_tokens_gen"]
        all_movie = case["all_movies"]

        es_idx = topic_path if topic_path != [] else [
            self.topic_class_num]
        es_idx, es_idx_len, es_idx_mask = clip_pad_sentence(es_idx, DO.max_t_topic, None, None, save_prefix=True)

        tgt_idx = target
        ed1_idx = self.get_related_movies(topic_path,int(DO.max_p_topic*TO.connected_e))
        ed2, ed2_idx = self.toca.get_top_k_predicted_topic(topic_path, p_topic=tgt_idx, k=DO.max_p_topic - len(ed1_idx),
                                                           use_p_topic=DO.negtive_sample,find_idx = ed1_idx)
        ed1_idx.extend(ed2_idx)
        ed_idx = ed1_idx[:DO.max_p_topic]
        sub_graph = self.epair_to_target(ed_idx, es_idx, es_idx_len)

        ed_idx = torch.tensor(ed_idx.copy(), dtype=torch.long)
        es_idx_mask = torch.tensor(es_idx_mask, dtype=torch.long)
        es_idx_len = torch.tensor(len(topic_path), dtype=torch.long)
        es_idx = torch.tensor(es_idx, dtype=torch.long)

        response_index, _, _ = clip_pad_sentence(response, DO.target_max_len_gen)
        response_index = torch.tensor(response_index, dtype=torch.long)

        context_gen, _, _ = clip_pad_sentence(context_gen[1:], DO.context_max_his_len)
        context_gen = torch.tensor(context_gen, dtype=torch.long)

        context_index = self.tokenizer.convert_tokens_to_ids(context[1:])

        q_context_index = self.tokenizer.tokenize(self.id_to_topic[target])
        q_context_index = self.tokenizer.convert_tokens_to_ids(q_context_index) + [DO.PreventWord.target_EOS_ID]

        q_context_index.extend(context_index)
        q_context_index, _, _ = clip_pad_sentence(q_context_index, DO.context_max_his_len_q)
        q_context_index = torch.tensor(q_context_index, dtype=torch.long)


        context_index, _, _ = clip_pad_sentence(context_index, DO.context_max_his_len)
        context_index = torch.tensor(context_index, dtype=torch.long)

        target_index = target
        target_index = torch.tensor(target_index, dtype=torch.long)



        batch = [all_movie,
                 identity,
                 response_index,
                 context_gen,
                 context_index,
                 target_index,
                 q_context_index,
                 es_idx, ed_idx, es_idx_len, es_idx_mask,sub_graph]
        return batch


    def __len__(self):
        return len(self.sessions)

if __name__ == '__main__':
    a = np.array([1, 6, 2, 8])
    b = np.array(range(100)).reshape((10,10))

    pairs = np.array(list(permutations(a,2)))
    sub=b[pairs[:,0],pairs[:,1]].reshape(a.shape[0],a.shape[0]-1)
    self_sub = np.expand_dims(b[a,a],axis=-1)
    sub = np.concatenate((self_sub,sub),axis=-1)
    sentence =[1,2,3,4,5,6]
    clip_pad_sentence(sentence,10
                          ,sos=DO.PreventWord.SOS_ID,
                          eos=DO.PreventWord.EOS_ID,
                          pad=DO.PreventWord.PAD_ID,
                          save_prefix=False)
