import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from resource.model.rec_model import rec_model
from resource.input.session_dataset import SessionDataset
from resource.option.train_option import TrainOption
from resource.option.dataset_option import DatasetOption
from resource.tensor_nl_interpreter import TensorNLInterpreter
from resource.util.file_util import mkdir_if_necessary
from resource.util.json_writer import JsonWriter
from resource.input.session_dataset import my_collate_fn
import  transformers
import json

engine_logger = logging.getLogger("main.engine")
torch.set_default_tensor_type(torch.FloatTensor)
class Rec_Engine:
    def __init__(self,
                 model: rec_model,
                 train_dataset: SessionDataset,
                 test_dataset: SessionDataset,
                 valid_dataset: SessionDataset,
                 graph=None,
                 edge_list=None,
                 toca=None,
                 topics_num=None
                 ):

        self.model = model
        edge_list = list(set(edge_list))
        self.edge_sets = torch.LongTensor(edge_list)
        self.edge_idx = self.edge_sets[:, :2].t()
        self.edge_type = self.edge_sets[:, 2]

        self.toca = toca
        self.graph = graph
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=TrainOption.train_batch_size,
                                           shuffle=True,
                                           collate_fn=lambda x:my_collate_fn(x),
                                           num_workers=TrainOption.data_load_worker_num,pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=TrainOption.test_batch_size,
                                          shuffle=False,
                                          collate_fn=lambda x:my_collate_fn(x),
                                          num_workers=TrainOption.data_load_worker_num,pin_memory=True)
        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=TrainOption.valid_batch_size,
                                           shuffle=False,
                                           collate_fn=lambda x:my_collate_fn(x),
                                           num_workers=TrainOption.data_load_worker_num,pin_memory=True)

        self.topics_num = topics_num
        self.tensor_nl_interpreter = TensorNLInterpreter(vocab=self.topics_num)
        self.json_writer = JsonWriter()
        self.init_optim()
        movie_ids_filename = DatasetOption.movie_ids if DatasetOption.dataset == "Redial" else DatasetOption.movie_ids_TG
        self.movie_ids = json.load(open(movie_ids_filename, 'r', encoding='utf-8'))


    def init_optim(self):
        param_optimizer = list(self.model.named_parameters())  # 模型参数名字列表
        encoder_name = ['topath_encoder', 'profile_encoder', 'his_encoder']
        graph_encoder_name = ['RGCN','topic_embedder', 'GCN']

        optimizer_bert_parameters = [
                p for n, p in param_optimizer
                if any(nd in n for nd in encoder_name)
            ]
        optimizer_graph_encoder_parameters = [
            p for n, p in param_optimizer
            if any(nd in n for nd in graph_encoder_name)
        ]
        optimizer_decoder_parameters = [
                p for n, p in param_optimizer
                if not any(nd in n for nd in encoder_name) and not any(nd in n for nd in graph_encoder_name)
            ]

        self.optimizer_decoder = optim.Adam(optimizer_decoder_parameters, lr=1e-3)
        self.optimizer_graph_encoder = optim.Adam(optimizer_graph_encoder_parameters, lr=5e-4)
        self.optimizer_bert = transformers.AdamW(optimizer_bert_parameters,lr=1e-5)
        self.optimizer = [self.optimizer_decoder, self.optimizer_graph_encoder, self.optimizer_bert]


    def train(self,pretrain=False):

        optim_interval = int(TrainOption.efficient_train_batch_size / TrainOption.train_batch_size)

        if pretrain:
            #********************************************stage1 for pretraining********************************************
            pretrain_best_metrics = [0.0]
            global_step = 0
            engine_logger.info("optim interval = {}".format(optim_interval))
            for epoch in range(TrainOption.epoch1):
                pbar = tqdm(self.train_dataloader)
                engine_logger.info("EPOCH {}".format(epoch))
                # FOR BATCH
                for batch_data in pbar:
                    subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                    [batch_data,identities ] = batch_data

                    sub_graph = batch_data[-1].to(TrainOption.device)
                    batch_data = [data.to(TrainOption.device) for data in batch_data[:-1]]
                    global_step += 1
                    do_optim = (global_step % optim_interval == 0)

                    # LOAD DATA
                    targets = batch_data[3]
                    outputs, p_relation, q_relation = self.model.forward(graph=subgraphs, inputs=batch_data,
                                                                         pretrain=True)
                    loss1 = self.model.loss(
                        outputs.reshape(-1, self.topics_num + 1),
                        targets.reshape(-1).detach())
                    loss2 = self.model.loss_KL(p_relation, q_relation.detach(), es_mask=batch_data[-1])
                    loss3 = self.model.loss_pretrain(p_relation, q_relation, batch_data[-1], sub_graph)
                    loss=loss1+ TrainOption.lambda_1*loss3
                    print("step:",global_step," NLL loss: {:.4f}".format(loss1.item()), " KL loss: {:.4f}".format(loss2.item()), " BCE loss: {:.4f}".format(loss3.item())
                          ,"KL loss is useless in pretrain process")
                    del outputs, p_relation, q_relation
                    # OPTIMIZATION
                    loss /= optim_interval
                    loss.backward()
                    loss_info = "loss: {:.4f}".format(loss.item())
                    pbar.set_description("TASK-ID: {}.log - ".format(TrainOption.task_uuid) + loss_info)
                    if do_optim:
                        for optimizer in self.optimizer:
                            optimizer.step()
                            optimizer.zero_grad()
                    # TEST
                    if global_step % TrainOption.test_eval_interval == 0:

                        # EVALUATION
                        pretrain_metrics = self.test(self.test_dataloader,pretrain = True)
                        pretrain_metrics[0] = pretrain_metrics[0] * 5
                        if pretrain_best_metrics is None or sum(pretrain_metrics) > sum(pretrain_best_metrics):
                            pretrain_best_metrics = pretrain_metrics
                            ckpt_filename = DatasetOption.ckpt_filename_template_pretrain.format(dataset=DatasetOption.dataset,
                                                                                        task=DatasetOption.task,
                                                                                        uuid=TrainOption.task_uuid,
                                                                                        global_step=global_step)
                            mkdir_if_necessary(os.path.dirname(ckpt_filename))
                            self.model.dump(ckpt_filename)
                            engine_logger.info("dump new best pretrain model checkpoint to {}\n".format(ckpt_filename))
                        else:
                            engine_logger.info("STEP {}, pretrain metrics are not improved".format(global_step))

                    # LOG LOSS INFO
                    if global_step % TrainOption.log_loss_interval == 0:
                        engine_logger.info("Pretrain STEP: {}, loss {}".format(global_step, loss_info))


        # ******************************************** stage2 for train ********************************************
        else:
            global_step = 0
            best_metrics = [0.0]
            o_best_metric_str = "(" + "-".join(["{:.4f}".format(x) for x in best_metrics]) + ")"
            engine_logger.info("optim interval = {}".format(optim_interval))
            for epoch in range(TrainOption.epoch2_rec):
                pbar = tqdm(self.train_dataloader)
                engine_logger.info("EPOCH {}".format(epoch))
                for batch_data in pbar:
                    subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                    [batch_data,identities ] = batch_data

                    sub_graph = batch_data[-1].to(TrainOption.device)
                    batch_data=[data.to(TrainOption.device) for data in batch_data[:-1]]
                    global_step += 1
                    do_optim = (global_step % optim_interval == 0)

                    # LOAD DATA
                    targets = batch_data[3]
                    outputs, p_relation, q_relation, gumbel_relation = self.model.forward(graph=subgraphs,inputs=batch_data,pretrain=False)
                    loss1 = self.model.loss(outputs.reshape(-1, self.topics_num+1), targets.reshape(-1).detach())
                    loss2 = self.model.loss_KL(p_relation, q_relation.detach() ,es_mask = batch_data[-1])
                    loss3 = self.model.loss_pretrain(p_relation, q_relation, batch_data[-1], sub_graph)
                    loss=loss1+TrainOption.lambda_0*loss2+ TrainOption.lambda_2*loss3
                    print("step:",global_step," NLL loss: {:.4f}".format(loss1.item()), " KL loss: {:.4f}".format(loss2.item()), " BCE loss: {:.4f}".format(loss3.item()))

                    del outputs, p_relation, q_relation

                    loss /= optim_interval
                    loss.backward()
                    loss_info = "loss: {:.4f}".format(loss.item())
                    pbar.set_description("TASK-ID: {}.log - ".format(TrainOption.task_uuid) + loss_info )

                    if do_optim:
                        for optimizer in self.optimizer:
                            optimizer.step()
                            optimizer.zero_grad()

                    # TEST
                    if global_step % TrainOption.test_eval_interval == 0:

                        # EVALUATION
                        metrics= self.test(self.test_dataloader)
                        #o_best_metric_str = "(" + "-".join(["{:.4f}".format(x) for x in best_metrics]) + ")"
                        metric_str = "(" + "-".join(["{:.4f}".format(x) for x in metrics]) + ")"

                        metrics[0] = metrics[0]*5

                        if best_metrics is None or sum(metrics) > sum(best_metrics):
                            engine_logger.info("STEP {}, origin metric: {}, best metric: {}".format(global_step,
                                                                                                    o_best_metric_str,
                                                                                                    metric_str))
                            best_metrics = metrics
                            o_best_metric_str = metric_str

                            ckpt_filename = DatasetOption.ckpt_filename_template.format(dataset= DatasetOption.dataset,
                                                                                        task = DatasetOption.task ,
                                                                                        uuid=TrainOption.task_uuid,
                                                                                        global_step=global_step,
                                                                                        metric=metric_str)
                            mkdir_if_necessary(os.path.dirname(ckpt_filename))
                            self.model.dump(ckpt_filename)
                            engine_logger.info("dump model checkpoint to {}\n".format(ckpt_filename))
                        else:
                            engine_logger.info("STEP {}, metrics are not improved, now is {}".format(global_step,metric_str))

                    if global_step % TrainOption.log_loss_interval == 0:
                        engine_logger.info("STEP: {}, loss {}".format(global_step, loss_info))

    def test(self, dataloader, mode="test" , pretrain = False):
        """test or valid"""
        assert mode == "test" or mode == "valid"
        self.model.eval()
        all_targets_index = []
        all_outputs_index = []
        engine_logger.info("{} START INFERENCE ...".format(mode.upper()))
        pbar = tqdm(dataloader)

        with torch.no_grad():
            for batch_data in pbar:
                pbar.set_description("TASK-ID: {}.log".format(TrainOption.task_uuid))

                # LOAD DATA
                subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                [ batch_data,identities] = batch_data

                batch_data = [data.to(TrainOption.device) for data in batch_data[:-1]]
                # LOAD DATA
                targets = batch_data[3]

                outputs = self.model.forward(graph=subgraphs, inputs=batch_data,
                                                          pretrain=pretrain)  # outputs:B,T   hit_cal:B,T,vocab_size

                all_outputs_index+=outputs.tolist()  #[all,T]
                all_targets_index+=targets.tolist()  #[all,T]
        engine_logger.info("{} INFERENCE FINISHED".format(mode.upper()))
        self.model.train()

        # metric
        metrics,cnt = self.eval_hit(all_targets_index, all_outputs_index)
        return metrics

    def eval_hit(self,targets,hit_cals):
        metrics_rec = {"recall@1":0, "recall@10":0,"recall@50":0, "loss": 0, "count": 0}

        cnt = 0
        hit_cals = torch.tensor(hit_cals)
        hit_cals = hit_cals[:, torch.LongTensor(self.movie_ids)]
        _, hit_cals = torch.topk(hit_cals, k=100, dim=1)
        for target,hit_cal,all_movie in zip(targets,hit_cals.tolist(),all_movie_list):
            tgt = target
            target_idx = self.movie_ids.index(tgt)
            metrics_rec["recall@1"] += int(target_idx in hit_cal[:1])
            metrics_rec["recall@10"] += int(target_idx in hit_cal[:10])
            metrics_rec["recall@50"] += int(target_idx in hit_cal[:50])
            metrics_rec["count"] += 1

        recall_1 = metrics_rec["recall@1"] / metrics_rec["count"]
        recall_10 = metrics_rec["recall@10"] / metrics_rec["count"]
        recall_50 = metrics_rec["recall@50"] / metrics_rec["count"]
        return [recall_1, recall_10, recall_50],cnt