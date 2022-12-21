import logging
import numpy as np
from resource.option.dataset_option import DatasetOption as DO
from tqdm import tqdm
import torch
import json
import re

vocab_logger = logging.getLogger("main.vocab")

def vocab_transfer(topic_num,topic_to_id,tokenizer):
    if DO.dataset == "Redial":
        dbid2entity = json.load(open(DO.DBID2en, 'r', encoding='utf-8'))  # {entity: entity_id}
        entity2dbid  = {entity:idx  for idx, entity in dbid2entity.items()}
        loc2glo = [0]*(topic_num+2)
        for topic,index  in tqdm(topic_to_id.items()):
            try:
                db_str = '@' + str(entity2dbid[topic])
                tok_id  = tokenizer[db_str]
                loc2glo[index] = tok_id
            except:
                db_str = topic.replace("<http://dbpedia.org/resource/","").replace(">","")
                try:
                    tok_id = tokenizer[db_str]
                    loc2glo[index] = tok_id
                except:
                    pass

    if DO.dataset=="TG":
        p = re.compile(r'[（](.*?)[）]', re.S)
        entity2id = json.load(open(DO.DBpedia2id_TG, 'r', encoding='utf-8'))  # {entity: entity_id}
        loc2glo = [0] * (topic_num + 2)
        for topic, index in tqdm(entity2id.items()):
            try:
                tok_id = tokenizer[re.sub(p,'',topic)]
                loc2glo[index] = tok_id
            except:
                pass


    return torch.tensor(loc2glo).cuda()