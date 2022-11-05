# -*- coding: utf-8 -*-


import os
from transformers import BertTokenizer

def mkdir_if_necessary(dir):
    if os.path.isdir(dir):
        return
    else:
        os.makedirs(dir)

def bert_tokenizer_redial():
    tokenizer = BertTokenizer(vocab_file="data/bert_base_Redial/vocab.txt")
    with open('data/redial/movie_dbid.txt','r') as f:
        dbid_tokens = f.read()
        dbid_tokens = dbid_tokens.split("\n")
        dbid_tokens = [id for id in dbid_tokens if id!=""]
    num_added_toks = tokenizer.add_tokens(dbid_tokens)
    print("add tokens for bert tokenizer in reidal dataset：",num_added_toks)
    return tokenizer