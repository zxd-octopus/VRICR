import numpy as np
from tqdm import tqdm
from collections import Counter
from resource.option.dataset_option import DatasetOption as DO
import json as js
import logging
import json

co_logger = logging.getLogger("main.co_appear")

class TOCoAppear:
    """DSCooccur
    topic Symptom CO-Occurrence, disease symptom co-occurrence probability via training data.

    existing topic --> predicting topic
    """

    def __init__(self, topic_to_id,topic_num = None):
        self.topic_to_id = topic_to_id
        self.all_topic = list(topic_to_id.keys())
        self.eps = 1e-24
        self.topic_num = len(self.all_topic) if topic_num is None else topic_num
        self.id_to_topic = {
            id: topic
            for topic, id in self.topic_to_id.items()
        }
        trans_matrix, topic2freq,topic2freq_list = self._get_trans_matrix()
        self.trans_matrix = trans_matrix
        self.topic2freq = topic2freq
        self.topic2freq_list= topic2freq_list
        self.topic2freq_mean = float(np.mean([x[1] for x in topic2freq.items()]))
        self.total = []
        self.cnt = []

    @staticmethod
    def _parse_trans(conv_topic_list):
        trans_pairs = list()

        for i in range(len(conv_topic_list)):
            main_role = conv_topic_list[i]
            next_index = len(conv_topic_list)
            trans_roles = conv_topic_list[i: next_index]
            for trans_role in trans_roles:
                trans_pairs.append((main_role, trans_role))

        return trans_pairs

    def _get_trans_matrix(self, freq_factor=0.2):
        """DSCoocur
        Returns:
            trans_matrix: given conv_topic, with trans matrix, predict next_topic_candidate
            topic_freq: a dict that key(topic(str)) and value(freq(int))
        """
        topic_trans_pairs = list()
        topics_counter = Counter()

        co_logger.info("processing entity CO-Occurrence via training data")
        if DO.dataset == "TG":
            entity2id = json.load(open(DO.DBpedia2id_TG, 'r', encoding='utf-8'))  # {entity: entity_id}
            with open(DO.raw_data_filename_TG.format("train"), 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            for conversation in tqdm(raw_data):
                dialog = conversation["messages"]
                conv_topic_list = []
                for utt in dialog:
                    movie_ids = [entity2id[movie] for movie in utt['movie'] if movie in entity2id]
                    entity_ids = [entity2id[entity] for entity in utt['entity'] if entity in entity2id]
                    conv_topic_list.extend(movie_ids)
                    conv_topic_list.extend(entity_ids)

                topics_counter.update(conv_topic_list)
                topic_trans_pairs += self._parse_trans(conv_topic_list)
# -----------------------------------------------------------------------------------------
        elif DO.dataset == "Redial":

            entity2id = json.load(open(DO.DBpedia2id, 'r', encoding='utf-8'))  # {entity: entity_id}
            with open(DO.raw_data_filename_Redial.format("train"), 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            for conversation in tqdm(raw_data):
                dialog = conversation["dialog"]
                conv_topic_list = []
                for utt in dialog:
                    movie_ids = [entity2id[movie] for movie in utt['movies'] if movie in entity2id]
                    entity_ids = [entity2id[entity] for entity in utt['entity'] if entity in entity2id]
                    conv_topic_list.extend(movie_ids)
                    conv_topic_list.extend(entity_ids)

                topics_counter.update(conv_topic_list)
                topic_trans_pairs += self._parse_trans(conv_topic_list)
#-----------------------------------------------------------------------------------------
        topic_ap_sum = sum(topics_counter.values())

        topic2freq = dict(list(topics_counter.most_common()))

        topic2freq_list = np.zeros(shape=self.topic_num)+ self.eps
        for i in topic2freq :
            topic2freq_list[i] += topic2freq[i] / topic_ap_sum


        # D, D
        trans_matrix = np.zeros(shape=(self.topic_num, self.topic_num))  # topic_num * topic_num

        for topic_pair in topic_trans_pairs:
            x, y = topic_pair[0], topic_pair[1]
            trans_matrix[x][y] += 1

        sum_matrix = np.sum(trans_matrix, axis=1)
        trans_matrix = trans_matrix / (
                    np.expand_dims(sum_matrix, axis=1) + self.eps) + self.eps

        return trans_matrix, topic2freq, topic2freq_list


    def get_top_k_predicted_topic(self,
                                     t_topic,
                                     p_topic=None,
                                     k=200,
                                    use_p_topic=False,
                                    find_idx = None):
        """get top k predicted topic
        Args:
            t_topic(list[str]): the topic appear in the history
            p_topic(list[idx]): the topic appear in the future
            k(int): default = 200
            use_p_topic： whether to ensure p_topic in selected_topics
        Returns:
            selected_topic(list[str]): a list of topic

        """

        def topic2index(t_topic):
            topic_vector = np.zeros(self.topic_num)
            if not isinstance(t_topic[0], int):
                topic_index = []
                for topic in t_topic:
                    t_idx = self.topic_to_id[topic]
                    topic_vector[t_idx] += 1
                    topic_index.append(t_idx)
            else:
                for topic in t_topic:
                    topic_vector[topic] += 1
                topic_index=t_topic

            return topic_vector, topic_index

        if len(t_topic) != 0:
            t_topic_vector, t_topic_index = topic2index(t_topic)  # [1, D] * [D, D] --> [1, D]
            topic2freq_ = np.expand_dims(self.topic2freq_list[t_topic_index], axis=-1)
            Gain = self.trans_matrix[t_topic_index] * topic2freq_ * \
                   np.log(self.trans_matrix[t_topic_index] / self.topic2freq_list)
            predict_vector = np.sum(Gain, axis=0)
        else:
            predict_vector = self.topic2freq_list
        top_k_idx = predict_vector.argsort()[::-1][0:k]

        def index2topic(index):
            topic = []
            for i in index:
                topic.append(self.id_to_topic[i] if i in self.id_to_topic.keys() else None)
            return topic

        if isinstance(p_topic,int):
            p_topic = [p_topic]
        for p in p_topic:
            if p not in top_k_idx:
                self.cnt.append(len(p_topic))

        if use_p_topic:
            p_topic_set = set(p_topic)
            top_k_idx = [d for d in top_k_idx if d not in p_topic_set]
            top_k_idx = p_topic + top_k_idx
            top_k_idx = top_k_idx[:k]
        candidate_topic = index2topic(top_k_idx)
        return candidate_topic,top_k_idx


if __name__=="__main__":
    topic_to_id = js.load(open(DO.topic_to_id_filename, 'r'))
    tocd=TOCoAppear(topic_to_id)
    tocd.get_top_k_predicted_topic(["笑话","升学","哲学"])
