import os
import uuid
import torch
from resource.option.dataset_option import DatasetOption


class TrainOption:
    lr = 1e-4
    mini_lr = 1e-5
    decay_rate = 0.9
    decay_interval = 5000
    #epoch1 = 100
    epoch1 = 15#pretrain
    epoch2_rec = 30
    epoch2_conv = 50
    train_batch_size = 20
    efficient_train_batch_size = 64
    test_batch_size = 128
    valid_batch_size = 128
    data_load_worker_num = 3
    scale_prj = True
    input_buffer_size = 100
    output_buffer_size = 100
    valid_eval_interval = 1000
    test_eval_interval = 1000
    log_loss_interval = 1000
    device = torch.device("cpu:0")
    task_uuid = str(uuid.uuid4())[:8]
    copynet = False
    warm_up_step = 5000
    decoder_layer = 12
    seprately = 1
    dec_con = False
    cp_ed = False
    rec_con = False
    rec_fusion = False
    ed_avg = False
    tgt_bert = False
    kg_network_fuison = False
    mask_mv = False
    mask_copy = False

    gradient_step_interval = 2
    without_t_know = False
    without_p_know = False
    without_copy = False
    attn_history_sentence = True
    attn_history_memory = True
    with_t_memory = True
    with_p_memory = True
    with_copy = True
    history_hop = 5
    consider_context_len = 10
    candidate_num = 200
    curtail_train = False
    beam_width = 1
    max_edge = 3
    add_posterior = False
    use_GCN=False
    use_RGCN = False
    init_bert_from_pretrain = True
    pretrain = False
    p_masked = 0.5
    copy = False
    GCN_tp = False
    mlp_fusion = False

    lambda_0 = 0.6
    lambda_1 = 0.0025
    lambda_2 = 0.0
    lambda_copy = 0.25
    loss_1 = True
    loss_2 = True
    loss_3 = True
    candidate_start_TG = 104
    candidate_start_Redial = 12
    connected_e = 0.6

    k = 0.0
    K = 5.0
    focal = False

    #graph_batch_size= int(train_batch_size * 512 / DatasetOption.sub_graph_num)

    loss_weight_floor = 0.2
    loss_weight_ceiling = 5
    with_gate = True
    copy_history = False


    bert_path = "data/bert_base_{}"
    origin_tau = 5
    mini_tau = 0.05
    tau_setp =  30000
    self_attention = False


    @staticmethod
    def update_device(device,device_id):
        if not device:
            return
        else:
            if device_id>0:
                os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(device_id)
            TrainOption.device = torch.device("cuda")

    @staticmethod
    def update_curtail_train(ct):
        TrainOption.curtail_train = ct

    @staticmethod
    def update_lr(lr):
        TrainOption.lr = lr

    @staticmethod
    def update(attr, value):
        setattr(TrainOption, attr, value)