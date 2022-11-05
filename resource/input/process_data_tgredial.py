from resource.option.dataset_option import DatasetOption as DO
import pickle
import logging
from tqdm import tqdm
import json
from collections import defaultdict


p_logger = logging.getLogger("main.input")


def process_topic_TGRedial():
    def filte_entity(entity):
        return entity.strip('<a>').strip('</').strip()

    entity_kg = open(DO.DBpedia_graph_TG, encoding='utf-8')
    entity2id = json.load(
        open(DO.DBpedia2id_TG, encoding='utf-8'))  # {entity: entity_id}
    id2entity = {idx: entity for entity, idx in entity2id.items()}


    n_entity = max(entity2id.values())
    p_logger.debug(
        f"[Load entity dictionary and KG from {DO.DBpedia2id} and {DO.DBpedia_graph}]")

    edge_list = []  # [(entity, entity, relation)]
    entity2neighbor = defaultdict(list)  # {entityId: List[entity]}
    for i, line in enumerate(entity_kg):
        triple = line.strip().split('\t')
        e0 = entity2id[triple[0]]
        e1 = entity2id[triple[2]]
        r = triple[1]
        edge_list.append((e0, e1, r))
        edge_list.append((e1, e0, r))
        edge_list.append((e0, e0, 'SELF_LOOP'))
        if e1 != e0:
            edge_list.append((e1, e1, 'SELF_LOOP'))

    relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if r not in relation2id:
            relation2id[r] = len(relation2id)
        edges.add((h, t, relation2id[r]))
        entities.add(id2entity[h])
        entities.add(id2entity[t])


    graph = defaultdict(list)
    edge_num = 0
    for item in edges:
        h,t,r = item
        if h != t:
            entity2neighbor[h].append(t)
        edge_num += 1
        graph[h].append(t)
    sum_len = 0
    max_len = 0
    e_num = 0
    for ed_list in graph.values():
        max_len = max(max_len, len(ed_list))
        sum_len += len(ed_list)
        e_num += 1
    p_logger.debug(f"[an entity can be connected to {max_len} entities at most]")
    p_logger.debug(f"[an entity can be connected to {sum_len/e_num} entities on average]")
    p_logger.info("External KGs process done！")

    topic2id = {filte_entity(entity):idx  for idx, entity in id2entity.items() if entity!='None'}

    graph_info=[topic2id,n_entity,relation2id,len(relation2id),graph,edges,entity2neighbor]
    return graph_info

def _load_other_data():
    # dbpedia
    entity2id = json.load(open(DO.DBpedia2id_TG, 'r', encoding='utf-8'))  # {entity: entity_id}
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    topic2ind = json.load(open(DO.topic2id, 'r', encoding='utf-8'))
    tok2ind = json.load(open(DO.tok2id_TG, 'r', encoding='utf-8'))
    ind2tok = {idx: word for word, idx in tok2ind.items()}
    return entity2id,id2entity,topic2ind,tok2ind,ind2tok

def get_dialog_info_TGReidal(tokenizer,task):
    entity2id, id2entity, topic2ind, tok2ind, ind2tok = _load_other_data()
    if DO.data_processed:
        with open(DO.save_filename_TG.format("train_"+task), 'rb+') as f:
            train = pickle.load(f)
        with open(DO.save_filename_TG.format("test_"+task), 'rb+') as f:
            test = pickle.load(f)
        with open(DO.save_filename_TG.format("valid_"+task), 'rb+') as f:
            valid = pickle.load(f)
        conv_info = [train,test,valid]
        return conv_info, tok2ind, ind2tok

    def _excute_data(subset,tokenizer):
        SEP = DO.PreventWord.SENTENCE_SPLITER
        SEP_GEN = DO.PreventWord.SENTENCE_SPLITER_ID_GEN
        PAD_ID = DO.PreventWord.PAD_ID
        augmented_convs_ =[]
        with open(DO.raw_data_filename_TG.format(subset), 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            p_logger.debug(f"[Load train data from {DO.raw_data_filename_Redial.format(subset)}]")
        for conversation in tqdm(raw_data):
            augmented_convs = []
            last_role = None
            conv_id = conversation['conv_id']
            for utt in conversation['messages']:
                local_id = utt['local_id']
                identity = str(conv_id)+'/'+str(local_id)
                assert utt['role'] != last_role
                text = utt["text"]
                text_token_gen = [tok2ind.get(word, DO.PreventWord.UNK_ID_GEN) for word in text]
                text_token_ids = tokenizer.tokenize(" ".join(text))
                movie_ids = [entity2id[movie] for movie in utt['movie'] if movie in entity2id]
                entity_ids = [entity2id[entity] for entity in utt['entity'] if entity in entity2id]
                policy = []
                for action, kw in zip(utt['target'][1::2], utt['target'][2::2]):
                    if kw is None or action == '推荐电影':
                        continue
                    if isinstance(kw, str):
                        kw = [kw]
                    kw = [topic2ind.get(k, PAD_ID) for k in kw]
                    policy.append([action, kw])
                final_kws = [topic2ind[kw] if kw is not None else PAD_ID for kw in utt['final'][1]]
                final = [utt['final'][0], final_kws]
                conv_utt_id = str(conversation['conv_id']) + '/' + str(utt['local_id'])

                augmented_convs.append({
                    "identity": identity,
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                    'policy': policy,
                    'final': final,
                    "text_gen" : text_token_gen,
                })
                last_role = utt["role"]
            augmented_convs_.append(augmented_convs)

        augmented_conv_dicts_ = []
        for raw_conv_dict in tqdm(augmented_convs_):
            augmented_conv_dicts = []
            context_tokens, context_entities, context_words, context_policy, context_items,context_tokens_gen = [], [], [], [], [],[]
            entity_set, word_set = set(), set()
            for i, conv in enumerate(raw_conv_dict):
                text_tokens, entities, movies, policies,text_tokens_gen = conv["text"], conv["entity"], conv["movie"], conv['policy'],conv["text_gen"]
                if len(context_tokens) > 0:
                    if task == "recommend":
                        for mv in movies:
                            conv_dict = {
                                "identity":conv['identity'],
                                "role": conv['role'],
                                "context_tokens": context_tokens.copy(),
                                "response": text_tokens_gen.copy(),
                                "context_entities": context_entities.copy(),
                                "items": mv,
                                "all_movies": movies.copy(),
                                "context_tokens_gen": context_tokens_gen.copy(),
                            }
                            augmented_conv_dicts.append(conv_dict)
                    if task == "generation":
                        conv_dict = {
                            "identity": conv['identity'],
                            "role": conv['role'],
                            "context_tokens": context_tokens.copy(),
                            "response": text_tokens_gen.copy(),
                            "context_entities": context_entities.copy(),
                            "items": 1,
                            "all_movies": movies.copy(),
                            "context_tokens_gen": context_tokens_gen.copy()
                        }
                        augmented_conv_dicts.append(conv_dict)


                context_tokens.extend([SEP]+text_tokens)
                context_tokens_gen.extend([SEP_GEN] + text_tokens_gen)
                context_policy.append(policies)
                context_items += movies
                for entity in entities + movies:
                    if entity not in entity_set:
                        entity_set.add(entity)
                        context_entities.append(entity)
                #pad_utters.append([0] * len(text_tokens))
            augmented_conv_dicts_.extend(augmented_conv_dicts)
        with open(DO.save_filename_TG.format(subset + "_" + task), 'wb') as f:
            pickle.dump(augmented_conv_dicts_, f)
        print(
            f"[Extract {len(augmented_conv_dicts_)} cases, from {DO.raw_data_filename_TG.format(subset)}]"
        )
        print(f"[Save processed data to {DO.save_filename_TG.format(subset)}]")
        return augmented_conv_dicts_


    train = _excute_data("train",tokenizer)
    test = _excute_data("test",tokenizer)
    valid = _excute_data("valid",tokenizer)
    conv_info = [train, test, valid]
    return conv_info, tok2ind, ind2tok

def process_data_TG(tokenizer,task):
    p_logger.info("Processing external KGs:")
    graph_info = process_topic_TGRedial()

    # process conversation
    p_logger.info("Processing dialogue session:")
    conv_info, tok2ind, ind2tok = get_dialog_info_TGReidal(tokenizer,task)
    return graph_info, conv_info, tok2ind, ind2tok

if __name__ == '__main__':
    pass