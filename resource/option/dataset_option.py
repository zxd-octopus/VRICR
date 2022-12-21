# -*- coding: utf-8 -*-


class DatasetOption:
    class PreventWord:
        PAD = "[PAD]"
        SOS = "[SOS]"
        EOS = "[EOS]"
        UNK = "[UNK]"
        SENTENCE_SPLITER = "[SEP]"
        WORD_SPLITER = "[wod]"
        mask="[MASK]"
        target_EOS = '[TGT]'

        PAD_ID = 0
        SOS_ID = 1
        EOS_ID = 2
        UNK_ID = 100
        UNK_ID_GEN = 3

        SENTENCE_SPLITER_ID = 102
        SENTENCE_SPLITER_ID_GEN = EOS_ID
        WORD_SPLITER_ID = 5
        mask_ID = 103
        target_EOS_ID = 7


    tgt_hyp_spliter = " <==> "
    test_filename_template = "data/cache/{dataset}/{task}/{uuid}/{mode}-{global_step}-{metric}.txt"
    ckpt_filename_template = "data/ckpt/{dataset}/{task}/{uuid}/{global_step}.model.ckpt"
    ckpt_filename_template_pretrain = "data/ckpt/{dataset}/{task}/{uuid}/pretrain-{global_step}.model.ckpt"

    dataset = "TG"
    task = "recommend"


    DBpedia2id_TG = 'data/TG_c2/entity2id.json'
    raw_data_filename_TG = "data/TG_c2/{}_data.json"
    save_filename_TG = "data/TG_c2/{}_processed_data_TG.pkl"
    movie_ids_TG = "data/TG_c2/movie_ids.json"
    DBpedia_graph_TG = "data/TG_c2/cn-dbpedia.txt"
    conceptnet_graph_TG = "data/TG_c2/hownet.txt"
    bert_vocab_zh = "data/bert_base_TG/vocab.txt"
    topic2id = "data/TG_c2/topic2id.json"
    tok2id_TG = "data/TG_c2/token2id.json"

    

    #Redial
    DBpedia2id = 'data/Redial_c2/entity2id.json'
    raw_data_filename_Redial = "data/Redial_c2/{}_data.json"
    save_filename_Redial = "data/Redial_c2/{}_processed_data_Redial.pkl"
    movie_ids = "data/Redial_c2/movie_ids.json"
    DBpedia_graph = "data/Redial_c2/dbpedia_subkg.json"
    conceptnet_graph = "data/Redial_c2/conceptnet_subkg.txt"
    bert_vocab_en = "data/bert_based_Redial/vocab.txt"
    DBID2en = "data/Redial_c2/id2entity.jsonl"
    tok2id = "data/Redial_c2/token2id.json"
    mid2name = 'data/Redial_c2/mid2name_redial.json'

    test_num = 3000
    valid_num = 3000
    vocab_size = 30000 #23963
    embed_dim = 300
    trans_embed_dim = 512
    kg_emb_dim = 128
    data_processed=False
    context_max_profile_len = 100
    context_max_his_len = 200
    context_max_his_len_q = context_max_his_len + 20
    context_max_path_len = 50
    target_max_len = 2
    target_max_len_Redial = 3
    max_t_topic=10
    max_p_topic=100
    max_negtive_topic = 500
    max_t_topic_TG_Rec = 75
    negtive_sample = True
    negtive_sample_num = 500
    target_max_len_gen = 50

    @staticmethod
    def update(attr, value):
        setattr(DatasetOption, attr, value)