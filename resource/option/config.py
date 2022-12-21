# -*- coding: utf-8 -*-
import logging
import argparse
from resource.option.train_option import TrainOption
from resource.option.dataset_option import DatasetOption
from resource.option.transformer_option import TransformerOption

cfg_logger = logging.getLogger("main.cfg")


def parse_cfg(args, cfgs):
    if cfgs is None:
        return

    for cfg in cfgs:
        k, v = cfg.split("=")

        try:
            v = int(v)  # parse to int
        except (Exception,):
            pass

        setattr(args, k, v)  # update args
        TrainOption.update(k, v)  # update TrainOption
    delattr(args, "cfg")


def config():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model",choices=["", "",],required=True)

    parser.add_argument("--dataset", default="TG", help="TG or Redial")
    parser.add_argument("--eval_interval",  type=int, default=500)
    parser.add_argument("--warm_up_step", type=int, default=8000)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--negtive", action="store_true", default=True)
    parser.add_argument("--task", default="TG", help="recommend or generation")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--use_GCN", action="store_true", help="use gpu", default=False)
    parser.add_argument("--use_RGCN", action="store_true", help="use gpu", default=True)
    parser.add_argument("--gpuid", type=int, default=-1)#<0
    parser.add_argument("--gpu", action="store_true", help="use gpu", default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("-tnl", "--tf_num_layers", type=int)
    parser.add_argument("--dev", action="store_true", help="on developing, choose the data version dev")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--gradient_step_interval", type=int, default=4)
    parser.add_argument("--cfg", nargs="*")
    parser.add_argument("-bw", "--beam_width", type=int, default=1)
    parser.add_argument("--rec_con", action="store_true", default=False)


    parser.add_argument("-l0", "--lambda_0", type=float, default=10.0)
    parser.add_argument("-l1", "--lambda_1", type=float, default=0.0025)
    parser.add_argument("-l2", "--lambda_2", type=float, default=0.0025)
    parser.add_argument("-l3", "--lambda_3", type=float, default=0.01)
    parser.add_argument("-fc", "--focal", action="store_true")
    parser.add_argument("-tid", "--task_uuid", type=str, default=None)
    parser.add_argument("--random_delete", type=float, default=0.0)
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt")
    parser.add_argument("--coarse_num",type=int,default=50)
    parser.add_argument("--fine_num",type=int,default=15)
    parser.add_argument("--data_processed", action="store_true",default=False)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--sub_graph_num",type=int,default=10)
    parser.add_argument("--tau_setp",type=int,default=4000)
    parser.add_argument("--decoder_layer", type=int, default=12)
    parser.add_argument("--efficient_train_batch_size", type=int, default=512)
    parser.add_argument("--seprately",  type=int, default=5)
    parser.add_argument("--copy", action="store_true", default=False)
    parser.add_argument("--dec_con", action="store_true", help="concacte relation and node representation for decoder", default=False)
    parser.add_argument("--cp_ed",action = "store_true",default=False)
    parser.add_argument("--self_attention", action="store_true", default=False)
    parser.add_argument("--GCN_tp", action="store_true", default=True)
    parser.add_argument("--mlp_fusion",action="store_true",default=False)
    parser.add_argument("--rec_fusion",action="store_true",default=False)
    parser.add_argument("--tgt_bert",action="store_true",default=True)

    args = parser.parse_args()

    # update TrainOption
    TrainOption.update_device(args.gpu,args.gpuid)
    TrainOption.update_lr(args.lr)

    TrainOption.update("lr", args.lr)
    TrainOption.update("rec_con", args.rec_con)
    TrainOption.update("use_GCN", args.use_GCN)
    TrainOption.update("use_RGCN", args.use_RGCN)
    TrainOption.update("train_batch_size", args.train_batch_size)
    TrainOption.update("test_batch_size", args.test_batch_size)
    TrainOption.update("valid_batch_size", args.test_batch_size)
    TrainOption.update("gradient_step_interval", args.gradient_step_interval)
    TrainOption.update("beam_width", args.beam_width)
    TrainOption.update("valid_eval_interval", args.eval_interval)
    TrainOption.update("test_eval_interval", args.eval_interval)
    TrainOption.update("pretrain",args.pretrain)
    TrainOption.update("tau_setp",args.tau_setp)
    TrainOption.update("decoder_layer", args.decoder_layer)
    TrainOption.update("efficient_train_batch_size", args.efficient_train_batch_size)
    TrainOption.update("seprately", args.seprately)
    TrainOption.update("dec_con", args.dec_con)
    TrainOption.update("copy", args.copy)
    TrainOption.update("cp_ed", args.cp_ed)
    TrainOption.update("self_attention", args.self_attention)
    TrainOption.update("GCN_tp", args.GCN_tp)
    TrainOption.update("mlp_fusion",args.mlp_fusion)
    TrainOption.update("rec_fusion",args.rec_fusion)
    TrainOption.update("tgt_bert",args.tgt_bert)



    TrainOption.update("lambda_0", args.lambda_0)
    TrainOption.update("lambda_1", args.lambda_1)
    TrainOption.update("lambda_2", args.lambda_2)
    TrainOption.update("lambda_copy", args.lambda_3)
    DatasetOption.update("data_processed", args.data_processed)
    DatasetOption.update("sub_graph_num", args.sub_graph_num)
    DatasetOption.update("dataset", args.dataset)
    DatasetOption.update("task", args.task)
    DatasetOption.update("max_p_topic",args.sample)
    DatasetOption.update("trans_embed_dim", args.embed_dim)
    if args.negtive:
        DatasetOption.update("negtive_sample",False)

    TransformerOption.update("n_warmup_steps",args.warm_up_step)


    if args.task_uuid is not None:
        TrainOption.update("task_uuid", args.task_uuid)

    parse_cfg(args, args.cfg)
    parameters = []
    parameter_template = "\n{:>30}: {}"
    parameter_board = "\n" + "=" * 50


    for item in sorted(args.__dict__.items(), key=lambda x: x[0]):
        if not item[0].startswith("__"):
            parameters.append(parameter_template.format(item[0], item[1]))
    parameter = parameter_board + "".join(parameters) + parameter_board
    cfg_logger.info(parameter)

    if args.dev:
        DatasetOption.raw_data_filename = DatasetOption.raw_data_filename + ".dev"
        TrainOption.data_load_worker_num = 1
        TrainOption.valid_eval_interval = 20
        TrainOption.test_eval_interval = 20

    return args
