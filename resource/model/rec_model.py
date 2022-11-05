from resource.model.base_model import BaseModel
import torch.nn as nn
import torch
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.module.transformer_models import get_pad_mask, get_subsequent_mask, Logit_Scale,repeat_if_not_none
from resource.module.transformer_SubLayers import SelfAttentiveEncoder,MultiHeadAttention,SelfAttentionSeq,SelfAttentionEd,SelfAttentionBatch
from resource.module.gumbel_sampling import GUMBEL
from transformers import BertModel
from einops import repeat
import torch.nn.functional as F
from resource.module.tau_scheduler import TauScheduler
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from resource.module.bs_misc import kg_context_infer_network

class rec_model(BaseModel):
    def __init__(self,
                 topics_num,
                 kg_emb_dim,
                 d_word_vec,
                 d_model,
                 d_inner,
                 n_head,
                 device):
        super(rec_model, self).__init__(device)

        self.pad_idx = DO.PreventWord.PAD_ID
        self.loss_p = nn.BCELoss(size_average=False, reduce=False)
        assert d_model == d_word_vec, \
            'To facilitate the residual connections, the dimensions of all module outputs shall be the same.'
        self.topics_num = topics_num+1
        self.relations_num = 2
        self.n_head=n_head
        self.kg_emb_dim = kg_emb_dim
        self.d_model = d_model
        self.d_inner = d_inner

        self._build_graph_encoder_layer()
        self._build_relation_infer_layer()
        self._build_encoder_layer()
        self._build_decoder_layer()

    def freeze_parameters(self, freeze_models):
        print('[freeze {} parameter unit]'.format(len(freeze_models)))
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False


    def _build_encoder_layer(self):
        self.path_att = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        self.his_encoder = BertModel.from_pretrained(TO.bert_path.format(DO.dataset))
        self.profile_encoder = BertModel.from_pretrained(TO.bert_path.format(DO.dataset))

        for model in [self.his_encoder, self.profile_encoder]:
            for param in model.parameters():
                param.requires_grad = True

    def _build_graph_encoder_layer(self):
        self.RGCN = RGCNConv(self.topics_num, self.kg_emb_dim, self.relations_num, num_bases=8)

    def _build_relation_infer_layer(self):
        self.conv_context_entity = nn.Linear(self.d_model, self.kg_emb_dim)
        self.conv_profile_entity = nn.Linear(self.d_model, self.kg_emb_dim)

        self.postrior = kg_context_infer_network(self.kg_emb_dim, self.d_inner, self.relations_num)
        self.prior = kg_context_infer_network(self.kg_emb_dim, self.d_inner, self.relations_num)

        self.softmax = nn.Softmax(dim=-1)

        self.relation_gumbel = GUMBEL(self.relations_num, self.d_inner, self.training, gumbel_act=True)
        self.ts = TauScheduler(TO.origin_tau, TO.mini_tau, TO.tau_setp)

    def _build_decoder_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.topics_num)

    def context_encoder(self,context_index, target=False, q_context_token_index = None):
        his_src_mask = get_pad_mask(context_index, self.pad_idx)
        q_enc_his_pooler = None
        enc_context_pooler = self.his_encoder(context_index, his_src_mask)['pooler_output']
        if target:
            q_context_token_index_mask = get_pad_mask(q_context_token_index, self.pad_idx)  # B, 1, T (torch.bool) for attention mask
            q_enc_his_pooler = self.his_encoder(q_context_token_index, q_context_token_index_mask)['pooler_output']

        return enc_context_pooler, q_enc_his_pooler

    def relation_cal(self,his,tgt_rep=None,es_idx=None,ed_idx=None, graph=None):

        edge_idx, edge_type = graph
        topic_rep = self.RGCN(None, edge_idx, edge_type)

        [bs, es_num] = list(es_idx.size())
        [bs, ed_num] = list(ed_idx.size())
        size = his.size()


        context_rep = his
        # ********************************************************posterior********************************************************
        if self.training:
            es_rep = torch.index_select(topic_rep, dim=0, index=es_idx.long().view(-1)).reshape(bs, es_num, -1)
            ed_rep = torch.index_select(topic_rep, dim=0, index=ed_idx.long().view(-1)).reshape(bs, ed_num,
                                                                                                -1)  # [bs,ed_num,dim]
            context_rep_ = self.conv_profile_entity(tgt_rep)
            context_rep_ = context_rep_.view(size[0], 1, 1, -1).repeat(1, ed_num, es_num, 1)

            ed_rep = repeat(ed_rep, 'i j k -> i j a k', a=es_num)
            es_rep = repeat(es_rep, 'i j k -> i a j k', a=ed_num)

            q_relation = self.postrior(es_rep, ed_rep, context_rep_)

            q_relation_fine = self.softmax(q_relation)

        # ********************************************************prior********************************************************
        es_rep = torch.index_select(topic_rep, dim=0, index=es_idx.long().view(-1)).reshape(bs, es_num, -1)
        ed_rep = torch.index_select(topic_rep, dim=0, index=ed_idx.long().view(-1)).reshape(bs, ed_num, -1)

        context_rep_ = self.conv_context_entity(context_rep)
        context_rep_ = context_rep_.view(size[0], 1, 1, -1).repeat(1, ed_num, es_num, 1)

        ed_rep = repeat(ed_rep, 'i j k -> i j a k', a=es_num)
        es_rep = repeat(es_rep, 'i j k -> i a j k', a=ed_num)

        p_relation = self.prior(es_rep, ed_rep, context_rep_)

        p_relation_fine = self.softmax(p_relation)

        if self.training :
            return p_relation_fine, q_relation_fine, q_relation, topic_rep

        else:
            return p_relation_fine, topic_rep


    def sample(self,hyp_relation,es_mask=None):

        hyp_relation = self.relation_gumbel(hyp_relation,self.training,self.ts.step_on())

        bs, ed_num, es_num, relation_num = hyp_relation.size()

        es_mask = repeat(es_mask, 'i j -> i k j', k=ed_num)
        ed_prob = hyp_relation[:,:,:,-1].masked_fill(es_mask == 0, 0)
        ed_prob = torch.einsum('ikj->ik', ed_prob)
        es_mask = 1 / es_mask.sum(-1).float()
        ed_prob = torch.mul(ed_prob, es_mask)

        return ed_prob,hyp_relation

    def _encode_user(self, entity_lists, es_lens, kg_embedding):
        user_repr_list = []
        user_attn_list = []
        entity_lists = entity_lists.tolist()
        es_lens = es_lens.tolist()
        for entity_list,es_len in zip(entity_lists, es_lens):
            mask_attn = torch.tensor((10-es_len)*[0],device=kg_embedding.device)
            if es_len == 0:
                user_repr_list.append(torch.zeros(self.kg_emb_dim, device=kg_embedding.device))
                user_attn_list.append(mask_attn)
                continue
            user_repr = kg_embedding[entity_list[:es_len]]
            user_repr, user_attn,_ = self.path_att(user_repr)
            user_repr_list.append(user_repr)
            user_attn_list.append(torch.cat((user_attn,mask_attn),0))
        return torch.stack(user_repr_list, dim=0) ,torch.stack(user_attn_list, dim=0)  # (bs, dim),(bs,es_num)

    def recommend(self, ed_prob = None, ed_idx = None ,topic_rep=None, es_idx=None,es_len = None, pretrain = False):

        if pretrain:
            es_attn_rep, es_attn = self._encode_user(es_idx, es_len, topic_rep)  # for es_rep
            user_rep = es_attn_rep

        else:
            ed_pre = topic_rep.index_select(0, ed_idx.view(-1)).reshape(ed_idx.size()[0], ed_idx.size()[1], -1)

            ed_attn_rep = torch.einsum('bk,bkd->bd', ed_prob, ed_pre)
            user_rep = ed_attn_rep

        rec_scores = F.linear(user_rep, topic_rep, self.rec_bias.bias)
        rec_scores = F.softmax(rec_scores)

        if not pretrain:
            copy_topic_temp = rec_scores.new_zeros(rec_scores.size(0), rec_scores.size(1))
            copy_topic_prob = copy_topic_temp.scatter_add(dim=1,
                                                          index=ed_idx,
                                                          src=ed_prob)

            ed_prob_avg = 1 / (copy_topic_prob.sum(-1) + 1e-10).unsqueeze(-1).detach()
            copy_topic_prob = torch.mul(copy_topic_prob, ed_prob_avg)

            rec_scores = 0.9 * rec_scores + 0.1 * copy_topic_prob

        return rec_scores

    def pretrain_forward(self, graph, inputs):
        [_,_,his_index, _, q_context_token_index, es_idx, ed_idx, es_idx_len,_] = inputs
        enc_context_pooler, q_enc_context_pooler\
            = self.context_encoder(his_index,target=True,q_context_token_index=q_context_token_index)
        enc_context_pooler = torch.randn(enc_context_pooler.size(), dtype=torch.float,
                                         device=enc_context_pooler.device)
        q_enc_context_pooler = torch.randn(enc_context_pooler.size(), dtype=torch.float,
                                   device=enc_context_pooler.device)

        p_relation, q_relation, q_relation_gumble, topic_rep \
            = self.relation_cal(his=enc_context_pooler, tgt_rep=q_enc_context_pooler,
                                es_idx=es_idx, ed_idx=ed_idx, graph=graph)
        outputs = self.recommend(topic_rep=topic_rep,
                                 es_idx=es_idx, es_len=es_idx_len ,pretrain = True)
        return outputs, p_relation, q_relation

    def train_forward(self,graph,inputs):
        [_,_,his_index,_,q_context_token_index,es_idx, ed_idx, es_idx_len, es_mask] = inputs

        enc_context_pooler, q_enc_context_pooler \
            = self.context_encoder(his_index, target=True, q_context_token_index= q_context_token_index)

        p_relation, q_relation,q_relation_gumble,topic_rep \
            =self.relation_cal(his=enc_context_pooler, tgt_rep=q_enc_context_pooler,
                                              es_idx=es_idx, ed_idx=ed_idx, graph =graph)

        ed_prob,gumbel_relation =  self.sample(q_relation_gumble,es_mask)
        outputs = self.recommend(ed_prob, ed_idx ,topic_rep=topic_rep, es_idx=es_idx, es_len=es_idx_len)
        return outputs, p_relation, q_relation,gumbel_relation

    def test_forward(self,graph, inputs,pretrain = False):
        [_,_,his_index, _,_, es_idx, ed_idx, es_idx_len, es_mask] = inputs
        enc_his_pooler ,_ = self.context_encoder(his_index,target=False)

        p_relation,topic_rep = self.relation_cal(his=enc_his_pooler,es_idx=es_idx, ed_idx=ed_idx, graph =graph)

        ed_prob,gumbel_relation = self.sample(p_relation, es_mask)
        if pretrain:
        # B * T, V
            outputs = self.recommend(topic_rep=topic_rep,es_idx=es_idx, es_len=es_idx_len, pretrain=True)
        else:
            outputs = self.recommend(ed_prob, ed_idx,
                                     topic_rep=topic_rep, es_idx=es_idx, es_len=es_idx_len)
        return outputs

    def forward(self,graph, inputs=None, pretrain=False):
        if self.training:
            if pretrain:
                return self.pretrain_forward(graph,inputs)
            else:
                return self.train_forward(graph,inputs)
        else:
            return self.test_forward(graph,inputs,pretrain)

    def loss_KL(self, p_relation, q_relation, es_mask):
        """
        p_relation  [bs,es_num,ed_num,edge_num]
        q_relation  [bs,es_num,ed_num,edge_num]
        es_mask     [bs,es_num]
        """

        bias = 1e-24
        kld_loss = nn.functional.kl_div(torch.log(p_relation + bias) + bias, q_relation, reduce=False).sum(
            -1)  # [bs,es_num,ed_num]

        es_mask = repeat(es_mask, 'b s -> b d s', d=p_relation.size(1))
        kld_loss = kld_loss.masked_fill(es_mask == 0, 0)
        es_mask = 1 / es_mask.sum(-1).unsqueeze(-1).float()
        kld_loss = torch.mul(kld_loss, es_mask)
        kld_loss = torch.sum(kld_loss) / kld_loss.size(0)

        return kld_loss

    def loss_pretrain(self, p_relation, q_relation, es_mask,sub_graph):
        """
        topic_pairs=[]
        graphs=[]
        for subgraph in subgraphs:
            [subtopic,subgraph,subconnection]=subgraph
            list(permutations(subtopic, 2))
            topic_pair=[i*self.topics_num+j for (i,j) in ]
        """
        target = sub_graph
        bias = 1e-24
        kld_loss_p = nn.functional.kl_div(torch.log(p_relation + bias) + bias, target, reduce=False).sum(
            -1)  # [bs,es_num,ed_num]
        kld_loss_q = nn.functional.kl_div(torch.log(q_relation + bias) + bias, target, reduce=False).sum(
            -1)  # [bs,es_num,ed_num]
        es_mask = repeat(es_mask, 'b s -> b d s', d=p_relation.size(1))

        kld_loss_q = kld_loss_q.masked_fill(es_mask == 0, 0)
        kld_loss_p = kld_loss_p.masked_fill(es_mask == 0, 0)
        es_mask = 1 / es_mask.sum(-1).unsqueeze(-1).float()
        kld_loss_q = torch.mul(kld_loss_q, es_mask)
        kld_loss_p = torch.mul(kld_loss_p, es_mask)

        kld_loss_p = torch.sum(kld_loss_p) / kld_loss_p.size(0)
        kld_loss_q = torch.sum(kld_loss_q) / kld_loss_q.size(0)

        return kld_loss_p + kld_loss_q

if __name__ == '__main__':
    total=torch.ones((2571,2571),dtype=torch.long).cuda()
    total_=torch.ones((2571,2571,5),dtype=torch.long).cuda()
    total__=torch.ones((16,20,200,44),dtype=torch.long).cuda
