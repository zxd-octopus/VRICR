
import logging
import torch.nn.functional as F

import math
import sys
import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from resource.input.session_dataset import SessionDataset
from resource.option.train_option import TrainOption
from resource.option.dataset_option import DatasetOption
from resource.tensor_nl_interpreter import TensorNLInterpreter
from resource.util.file_util import mkdir_if_necessary
from resource.util.json_writer import JsonWriter
from resource.module.scheduled_optim import ScheduledOptim
from resource.input.session_dataset import my_collate_fn
from  resource.util.distinct_redial import cal_calculate,cal_rouge
conv_engine_logger = logging.getLogger("main.conv_engine")
torch.set_default_tensor_type(torch.FloatTensor)
class Conv_Engine():
    def __init__(self,model:torch.nn.Module,
                 train_dataset: SessionDataset,
                 test_dataset: SessionDataset,
                 valid_dataset: SessionDataset,
                 d_model = None,
                 n_warmup_steps = 2000,
                 edge_list=None,
                 topics_num=None,
                 tokenizer = None,
                 lr=5e-5):
        self.model = model
        self.lr = lr
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = ScheduledOptim(self.optimizer, 0.5, d_model, n_warmup_steps)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=TrainOption.train_batch_size,
                                           shuffle=True,
                                           collate_fn=lambda x:my_collate_fn(x),
                                           num_workers=TrainOption.data_load_worker_num,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=TrainOption.test_batch_size,
                                          shuffle=False,
                                          collate_fn=lambda x:my_collate_fn(x),
                                          num_workers=TrainOption.data_load_worker_num,
                                          pin_memory=True)
        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=TrainOption.valid_batch_size,
                                           shuffle=False,
                                           collate_fn=lambda x:my_collate_fn(x),
                                           num_workers=TrainOption.data_load_worker_num,
                                           pin_memory=True)
        self.topics_num = topics_num
        self.tensor_nl_interpreter = TensorNLInterpreter(vocab=self.tokenizer)
        self.json_writer = JsonWriter()
        edge_list = list(set(edge_list))
        self.edge_sets = torch.LongTensor(edge_list)
        self.edge_idx = self.edge_sets[:, :2].t()
        self.edge_type = self.edge_sets[:, 2]

    def train(self,pretrian = None):
        global_step = 0
        best_metrics = [0.0] * 4
        best_metrics_valid = [0.0] * 4
        optim_interval = int(TrainOption.efficient_train_batch_size / TrainOption.train_batch_size)
        conv_engine_logger.info("optim interval = {}".format(optim_interval))
        for epoch in range(TrainOption.epoch_conv):
            pbar = tqdm(self.train_dataloader)
            conv_engine_logger.info("EPOCH {}".format(epoch))
            for batch_data in pbar:
                if TrainOption.use_RGCN:
                    subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                elif TrainOption.use_GCN:
                    subgraphs = self.edge_type.to(TrainOption.device)
                [batch_data,all_movies,identities ] = batch_data

                batch_data = [data.to(TrainOption.device) for data in batch_data[:-1]]
                global_step += 1
                do_optim = (global_step % optim_interval == 0)

                # LOAD DATA
                resp = batch_data[0]
                resp_gen = self.model.forward(graph=subgraphs,inputs=batch_data)

                loss, _ = nll_loss(resp_gen, resp.detach(), DatasetOption.PreventWord.PAD_ID)
                loss_info = "loss: {:.4f}".format(loss.item())
                pbar.set_description("TASK-ID: {}.log - ".format(TrainOption.task_uuid) + loss_info)
                loss = loss / optim_interval
                loss.backward(retain_graph=False)
                if do_optim:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # LOG LOSS INFO
                if global_step % TrainOption.log_loss_interval == 0:
                    conv_engine_logger.info("STEP: {}, loss {}".format(global_step, loss_info))

            # EVALUATION
            # valid
            all_targets_valid, all_outputs_valid, metrics_valid, all_identities_valid = self.test(self.valid_dataloader,mode="valid")
            metric_str_valid = "(" + "-".join(["{:.3f}".format(x) for x in metrics_valid][2:]) + ")"

            if best_metrics_valid is None or sum(metrics_valid) > sum(best_metrics_valid):
                best_metrics_valid = metrics_valid

                valid_filename = DatasetOption.test_filename_template.format(dataset=DatasetOption.dataset,
                                                                            task=DatasetOption.task,
                                                                            uuid=TrainOption.task_uuid,
                                                                            mode="valid",
                                                                            global_step=global_step,
                                                                            metric=metric_str_valid)
                mkdir_if_necessary(os.path.dirname(valid_filename))
                self.json_writer.write2file(filename=valid_filename,
                                            gths=all_targets_valid,
                                            hyps=all_outputs_valid,
                                            identites=all_identities_valid)

            conv_engine_logger.info(
                "STEP {}, Epoch {}, metric:dist@3-dist@4-rouge@1-rouge@2-rouge@l: {}".format(global_step, epoch,
                                                                                             metric_str_valid))

            #test
            all_targets, all_outputs, metrics, all_identities = self.test(self.test_dataloader)
            metric_str = "(" + "-".join(["{:.3f}".format(x) for x in metrics[2:]]) + ")"

            if best_metrics is None or sum(metrics) > sum(best_metrics):
                best_metrics = metrics

                test_filename = DatasetOption.test_filename_template.format(dataset=DatasetOption.dataset,
                                                                            task=DatasetOption.task,
                                                                            uuid=TrainOption.task_uuid,
                                                                            mode="test",
                                                                            global_step=global_step,
                                                                            metric=metric_str)
                mkdir_if_necessary(os.path.dirname(test_filename))
                self.json_writer.write2file(filename=test_filename,
                                            gths=all_targets,
                                            hyps=all_outputs,
                                            identites=all_identities)
            conv_engine_logger.info(
                "STEP {}, Epoch {}, metric:dist@3-dist@4-rouge@1-rouge@2-rouge@l: {}\n".format(global_step, epoch,
                                                                                               metric_str))


    def test(self,dataloader,mode="test"):
        assert mode in ["test","valid"]
        res_gen = []
        identity_list = []
        res_gth = []
        self.model.eval()

        conv_engine_logger.info("{} START INFERENCE ...".format(mode.upper()))
        pbar = tqdm(dataloader)

        with torch.no_grad():
            for batch_data in pbar:
                if TrainOption.use_RGCN:
                    subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                elif TrainOption.use_GCN:
                    subgraphs = self.edge_type.to(TrainOption.device)
                [batch_data,all_movies,identity ] = batch_data

                batch_data = [data.to(TrainOption.device) for data in batch_data[:-1]]

                # LOAD DATA
                resp = batch_data[0]
                resp_gen,probs= self.model.forward(graph=subgraphs, inputs=batch_data)
                resp_gen_word = self.tensor_nl_interpreter.interpret_tensor2nl(resp_gen)
                resp_gth_word = self.tensor_nl_interpreter.interpret_tensor2nl(resp)
                res_gen.extend(resp_gen_word)
                res_gth.extend(resp_gth_word)
                identity_list.extend(identity)
            dist_1, dist_2,dist_3,dist_4 = cal_calculate(res_gen)
            rouge1,rouge2,rouge_l = cal_rouge(res_gen,res_gth,DatasetOption.dataset,identity_list)
            print("dist_1:{},dist_2:{},dist_3:{},dist_4:{}".format(dist_1,dist_2,dist_3,dist_4))
            print("rouge1:{},rouge2:{},rouge_l:{}".format(rouge1, rouge2, rouge_l))
            sys.stdout.flush()
        self.model.train()
        conv_engine_logger.info("{} INFERENCE FINISHED".format(mode.upper()))
        return res_gth, res_gen,[dist_1, dist_2,dist_3,dist_4,rouge1,rouge2,rouge_l],identity_list


def nll_loss(hypothesis, target, pad_id ):
        eps = 1e-9
        B, T = target.shape
        hypothesis = hypothesis.reshape(-1, hypothesis.size(-1))
        target = target[:,1:]
        padding = torch.ones(target.size(0),1,dtype=torch.long) * pad_id
        padding = padding.cuda()
        target = torch.cat([target,padding],1)
        target = target.reshape(-1)
        nll_loss = F.nll_loss(torch.log(hypothesis + 1e-20), target, ignore_index=DatasetOption.PreventWord.PAD_ID, reduce=False)
        not_ignore_tag = (target != pad_id).float()
        not_ignore_num = not_ignore_tag.reshape(B, T).sum(-1)
        sum_nll_loss = nll_loss.reshape(B, T).sum(-1)
        nll_loss_vector = sum_nll_loss / (not_ignore_num + eps)
        nll_loss = nll_loss_vector.mean()
        return nll_loss, nll_loss_vector.detach()