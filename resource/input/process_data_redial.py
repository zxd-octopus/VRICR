from resource.option.dataset_option import DatasetOption as DO
import pickle
import logging
from tqdm import tqdm
import json
from collections import defaultdict


p_logger = logging.getLogger("main.input")


def process_topic_Redial(SELF_LOOP_ID=185):

    kg = json.load(open(DO.DBpedia_graph, 'r', encoding='utf-8'))
    topic2id = json.load(
        open(DO.DBpedia2id, 'r', encoding='utf-8'))  # {entity: entity_id}
    id2entity = {idx: entity for entity, idx in topic2id.items()}

    n_entity = len(topic2id)
    p_logger.debug(
        f"[Load entity dictionary and KG from {DO.DBpedia2id} and {DO.DBpedia_graph}]")

    edge_list = []
    entity2neighbor = defaultdict(list)  # {entityId: List[entity]}

    for entity in range(n_entity+1):
        edge_list.append((entity, entity, SELF_LOOP_ID))
        if str(entity) not in kg:
            continue
        for tail_and_relation in kg[str(entity)]:
            if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:  # and tail_and_relation[0] in EDGE_TYPES:
                edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h,t, r  in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)+1
    edge_list = [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000]

    graph = defaultdict(list)
    edge_num = 0
    for item in edge_list:
        h,t,r = item
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
    p_logger.info("External KGs process done!")
    topic2id = {entity:idx  for idx, entity in id2entity.items() if entity != 'None'}
    graph_info=[topic2id,n_entity,relation_idx,len(relation_idx),graph,edge_list,entity2neighbor]
    return graph_info

def _load_other_data():
    # dbpedia
    entity2id = json.load(open(DO.DBpedia2id, 'r', encoding='utf-8'))  # {entity: entity_id}
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    dbid2entity = json.load(open(DO.DBID2en, 'r', encoding='utf-8'))  # {entity: entity_id}
    tok2ind = json.load(open(DO.tok2id, 'r', encoding='utf-8'))
    ind2tok = {idx: word for word, idx in tok2ind.items()}
    return entity2id,id2entity,dbid2entity,tok2ind,ind2tok

def get_dialog_info_Reidal(tokenizer,task):

    entity2id, id2entity, dbid2entity, tok2ind, ind2tok = _load_other_data()

    if DO.data_processed:
        with open(DO.save_filename_Redial.format("train_"+task), 'rb+') as f:
            train = pickle.load(f)
        with open(DO.save_filename_Redial.format("test_"+task), 'rb+') as f:
            test = pickle.load(f)
        with open(DO.save_filename_Redial.format("valid_"+task), 'rb+') as f:
            valid = pickle.load(f)
        conv_info = [train,test,valid]
        return conv_info, tok2ind, ind2tok



    def _excute_data(subset,tokenizer):
        SEP = DO.PreventWord.SENTENCE_SPLITER
        SEP_GEN= DO.PreventWord.SENTENCE_SPLITER_ID_GEN
        augmented_convs_ =[]
        with open(DO.raw_data_filename_Redial.format(subset), 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            p_logger.debug(f"[Load {subset} data from {DO.raw_data_filename_Redial.format(subset)}]")
        for conversation in tqdm(raw_data):
            dialog = conversation["dialog"]
            augmented_convs = []
            last_role = None
            conv_id = conversation['conv_id']
            for utt in dialog:
                local_id = utt['utt_id']
                identity = str(conv_id)+'/'+str(local_id)
                text = utt["text"]
                text_token_gen = [tok2ind.get(word, DO.PreventWord.UNK_ID_GEN) for word in text]
                text_token = tokenizer.tokenize(" ".join(text))
                movie_ids = [entity2id[movie] for movie in utt['movies'] if movie in entity2id]
                entity_ids = [entity2id[entity] for entity in utt['entity'] if entity in entity2id]
                if utt["role"] == last_role:
                    augmented_convs[-1]["text"] += text_token
                    augmented_convs[-1]["movie"] += movie_ids
                    augmented_convs[-1]["entity"] += entity_ids
                    augmented_convs[-1]["text_gen"] += text_token_gen
                else:
                    augmented_convs.append({
                        "identity": identity,
                        "role": utt["role"],
                        "text": text_token,  # [utter_len]
                        "entity": entity_ids,
                        "movie": movie_ids,
                        "text_gen" : text_token_gen,
                    })
                last_role = utt["role"]
            augmented_convs_.append(augmented_convs)

        augmented_conv_dicts_ = []
        for raw_conv_dict in tqdm(augmented_convs_):
            augmented_conv_dicts = []
            context_tokens, context_entities, context_tokens_gen= [], [], []
            entity_set = set()
            for i, conv in enumerate(raw_conv_dict):
                text_tokens, entities, movies ,text_tokens_gen = conv["text"], conv["entity"], conv["movie"],conv["text_gen"]
                if len(context_tokens) > 0:
                    if task == "recommend":
                        for mv in movies:
                            conv_dict = {
                                "identity": conv['identity'],
                                "role": conv['role'],
                                "context_tokens": context_tokens.copy(),
                                "response": text_tokens_gen.copy(),
                                "context_entities": context_entities.copy(),
                                "items": mv,
                                "all_movies": movies.copy(),
                                "context_tokens_gen" : context_tokens_gen.copy(),
                                "response_word":text_tokens.copy(),
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
                context_tokens_gen.extend([SEP_GEN]+text_tokens_gen)
                for entity in entities + movies:
                    if entity not in entity_set:
                        entity_set.add(entity)
                        context_entities.append(entity)
            augmented_conv_dicts_.extend(augmented_conv_dicts)
        with open(DO.save_filename_Redial.format(subset+"_"+task), 'wb') as f:
            pickle.dump(augmented_conv_dicts_, f)
        print(
            f"[Extract {len(augmented_conv_dicts_)} cases, from {DO.raw_data_filename_Redial.format(subset)}]"
        )
        print(f"[Save processed data to {DO.save_filename_Redial.format(subset)}]")
        return augmented_conv_dicts_


    train = _excute_data("train",tokenizer)
    test = _excute_data("test",tokenizer)
    valid = _excute_data("valid",tokenizer)
    conv_info = [train, test, valid]
    return conv_info,tok2ind,ind2tok

def process_data_Redial(tokenizer,task):
    p_logger.info("Processing external KGs")
    graph_info = process_topic_Redial()

    # process conversation
    p_logger.info("Processing dialogue session")
    conv_info,tok2ind,ind2tok = get_dialog_info_Reidal(tokenizer,task)
    return graph_info, conv_info,tok2ind,ind2tok
