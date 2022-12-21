import os
import torch
import random
import numpy as np
from resource.util.get_logger import get_logger
from resource.option.train_option import TrainOption
from resource.option.config import config
from resource.input.process_data_redial import process_data_Redial
from resource.input.process_data_tgredial import process_data_TG
from resource.option.dataset_option import DatasetOption as DO
from resource.input.session_dataset import SessionDataset
from resource.model.rec_model import rec_model
from resource.option.transformer_option import TransformerOption as TRO
from transformers import BertTokenizer
from resource.rec_engine import Rec_Engine
from resource.conv_engine import  Conv_Engine
from resource.input.topic_co_appear import TOCoAppear
from resource.util.file_util import mkdir_if_necessary
from resource.util.json_writer import JsonWriter
from resource.input.vocab import vocab_transfer
from resource.model.conv_model import conversation_model
import gc


main_logger = get_logger("main", "data/log/{}.log".format(TrainOption.task_uuid))
main_logger.info("TASK ID {}".format(TrainOption.task_uuid))

def main():
    sed_num = 123
    random.seed(sed_num)
    np.random.seed(sed_num)
    torch.random.manual_seed(sed_num)
    args = config()
    main_logger.info("process data")

    assert DO.dataset in["TG","Redial"] and DO.task in ["recommend","generation"]

    tokenizer = None
    return_items = None
    if  DO.dataset == "Redial":
        tokenizer = BertTokenizer(vocab_file="data/bert_base_Redial/vocab.txt")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<http://dbpedia.org/resource/',"[wod]"]})
        return_items = process_data_Redial(tokenizer,DO.task)
    if DO.dataset == "TG":
        tokenizer = BertTokenizer(vocab_file="data/bert_base_TG/vocab.txt")
        tokenizer.add_special_tokens({'additional_special_tokens': ["[wod]"]})
        return_items = process_data_TG(tokenizer,DO.task)

    graph_info, conv_info, tok2ind,ind2tok = return_items
    topic_to_id, topic_class_num, relation_to_id, relation_class_num, graph, edge_list,entity2neighbor = graph_info

    train_sessions, test_sessions, valid_sessions=conv_info
    toca = TOCoAppear(topic_to_id,topic_num  = topic_class_num)

    train_dataset = SessionDataset(train_sessions,topic_to_id,tokenizer,toca, "train",relation_class_num,graph,entity2neighbor,topic_num=topic_class_num)
    test_dataset = SessionDataset(test_sessions,topic_to_id,tokenizer,toca, "test",relation_class_num,graph,entity2neighbor,topic_num=topic_class_num)
    valid_dataset = SessionDataset(valid_sessions,topic_to_id,tokenizer,toca, "valid",relation_class_num,graph,entity2neighbor,topic_num=topic_class_num)

    vocab_size = len(tok2ind)
    main_logger.info("creating model")

    # PROCESS MODEL
    model = None
    engine = None

    if not args.pretrain and args.ckpt is None:
        main_logger.warning("args.ckpt should not be None if not in pretrian stage")

    if DO.task == "recommend":
        model = rec_model(topics_num=topic_class_num,
                         kg_emb_dim=DO.kg_emb_dim,
                         d_word_vec=DO.trans_embed_dim,
                         d_model=DO.trans_embed_dim,
                         d_inner=TRO.dimension_hidden,
                         n_head=TRO.num_head,
                         device=TrainOption.device
                         )
        if args.ckpt is not None:
            model.load(args.ckpt, strict=False)
            main_logger.info("load weight from {}".format(args.ckpt))


    if DO.task == "generation":
        loc2glo = vocab_transfer(topic_class_num, topic_to_id, tok2ind)
        model = conversation_model(
            topics_num=topic_class_num,
            kg_emb_dim=DO.kg_emb_dim,
            d_word_vec=DO.trans_embed_dim,
            d_model=DO.trans_embed_dim,
            d_inner=TRO.dimension_hidden,
            n_head=TRO.num_head,
            device=TrainOption.device,
            loc2glo=loc2glo,
            vocab_size=vocab_size,
            d_k=TRO.dimension_key,
            d_v=TRO.dimension_key
        )
        if args.ckpt is not None:
            model.load(args.ckpt, strict=False)
            main_logger.info("load weight from {}".format(args.ckpt))

    model = model.to(TrainOption.device)
    if DO.task == "recommend":
        engine = Rec_Engine(model,
                                 train_dataset,
                                 test_dataset,
                                 valid_dataset,
                                 toca=toca,
                                 graph=graph,
                                 edge_list=edge_list,
                                 topics_num=topic_class_num
                            )
    if DO.task == "generation":
        engine = Conv_Engine(model,
                                     train_dataset,
                                     test_dataset,
                                     valid_dataset,
                                     d_model=DO.trans_embed_dim,
                                     tokenizer=ind2tok,
                                     edge_list=edge_list,
                                     topics_num=topic_class_num)

    gc.collect()

    if not args.test:
        engine.train(args.pretrain)

    else:
        # INFERENCE
        outputs = engine.test(engine.test_dataloader)#test
        if DO.task == "recommend":
            metrics = outputs
            metric = "(" + "-".join(["{:.3f}".format(x) for x in metrics]) + ")"
            main_logger.info("metric for recommendation: recall@1-recall@10-recall@50:{}".format(metric))

        if DO.task == "generation":
            res_gth, res_gen, metrics, identities_list = outputs

            metric = "(" + "-".join(["{:.3f}".format(x) for x in metrics[2:]]) + ")"#移除Dist1和Dist2的值
            main_logger.info("metric for generation:dist@3-dist@4-rouge@1-rouge@2-rouge@l:{}".format(metric))
            test_filename = DO.test_filename_template.format(dataset=DO.dataset,
                                                        task=DO.task,
                                                        uuid=TrainOption.task_uuid,
                                                        mode="test",
                                                        global_step="test", metric=metric)

            mkdir_if_necessary(os.path.dirname(test_filename))
            jsw = JsonWriter()

            jsw.write2file(filename=test_filename,
                           gths=res_gth,
                           hyps=res_gen,
                           identites = identities_list
                           )

if __name__ == '__main__':
    main()
