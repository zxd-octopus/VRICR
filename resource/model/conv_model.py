from resource.model.base_model import BaseModel
from resource.module.transformer_models import Decoder
import torch.nn as nn
import torch
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.module.transformer_models import get_pad_mask, get_subsequent_mask
from resource.module.gumbel_sampling import GUMBEL
from transformers import BertModel
from einops import repeat
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from resource.module.bs_misc import kg_context_infer_network
from resource.option.transformer_option import TransformerOption as TRO
from resource.module.transformer_models import Encoder

class conversation_model(BaseModel):
    def __init__(self,
                 topics_num,
                 kg_emb_dim,
                 d_word_vec,
                 d_model,
                 d_inner,
                 n_head,
                 device,
                 loc2glo,
                 vocab_size,
                 d_k = 64,
                 d_v = 64):
        super(conversation_model, self).__init__(device)

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
        self.loc2glo = loc2glo

        self.n_vocab = vocab_size
        self.d_word_vec = d_word_vec
        self.n_layers = TRO.num_layers
        self.n_head = TRO.num_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = TRO.dropout
        self.word_emb = nn.Embedding(self.n_vocab,d_word_vec,padding_idx=DO.PreventWord.PAD_ID)
        self.hidden_size = d_word_vec

        self.pad_idx = DO.PreventWord.PAD_ID
        self.bos_idx = DO.PreventWord.SOS_ID
        self.eos_idx = DO.PreventWord.EOS_ID

        self._build_encoder_layer()
        self._build_graph_infer_layer()
        self._build_decoder_layer()

    def _build_encoder_layer(self):
        self.tfr_encoder = Encoder(
            n_src_vocab=self.n_vocab, n_position=DO.context_max_his_len,
            d_word_vec=self.d_word_vec, d_model=self.d_model, d_inner=self.d_inner,
            n_layers=self.n_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v,
            pad_idx=self.pad_idx, dropout=self.dropout, scale_emb=False,
            word_emb=self.word_emb
        )

        self.conv_RGCN = RGCNConv(self.topics_num, self.kg_emb_dim, self.relations_num, num_bases=8)

    def freeze_parameters(self,freeze_models):
        print('[freeze {} parameter unit]'.format(len(freeze_models)))
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False


    def _build_graph_infer_layer(self):
        self.his_encoder = BertModel.from_pretrained(TO.bert_path.format(DO.dataset))
        self.conv_context_entity = nn.Linear(self.d_model, self.kg_emb_dim)
        self.RGCN = RGCNConv(self.topics_num, self.kg_emb_dim, self.relations_num, num_bases=8)
        self.prior = kg_context_infer_network(self.kg_emb_dim, self.d_inner, self.relations_num)
        self.relation_gumbel = GUMBEL(self.relations_num, self.d_inner, self.training, gumbel_act=True)
        graph_infer_modules = [self.his_encoder,self.conv_context_entity,self.RGCN,self.prior,self.relation_gumbel]
        self.freeze_parameters(graph_infer_modules)
        self.softmax = nn.Softmax(dim=-1)



    def _build_decoder_layer(self):
        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.d_word_vec)
        self.decoder = Decoder(
            n_trg_vocab=self.n_vocab, n_position=DO.target_max_len_gen,
            d_word_vec=self.d_word_vec, d_model=self.d_model, d_inner=self.d_inner,
            n_layers=self.n_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v,
            pad_idx=DO.PreventWord.PAD_ID, dropout=self.dropout, scale_emb=False,
            word_emb=self.word_emb
        )

        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size, self.n_vocab))

    def _encode_user_ed(self, ed_probs, es_lens, ed_pres, ed_idxs):  # 选出来所有ed的信息
        ed_repr_list = []
        ed_mask_list = []
        for i in range(ed_pres.size(0)):
            ed_prob, es_len, ed_pre, ed_idx = ed_probs[i, :], es_lens[i], ed_pres[i, :, :], ed_idxs[i, :]
            ed_idx_ = torch.nonzero(ed_prob).view(-1)
            ed_num = len(ed_idx_)
            mask_num = ed_pre.size(0) - ed_num
            ed_rep = torch.index_select(ed_pre, dim=0, index=ed_idx_)
            ed_rep = torch.cat([ed_rep, torch.zeros(mask_num, DO.kg_emb_dim, device=ed_pres.device)], dim=0)
            ed_mask = torch.cat(
                [torch.ones(ed_num, device=ed_pres.device), torch.zeros(mask_num, device=ed_pres.device)], dim=0)
            ed_repr_list.append(ed_rep)
            ed_mask_list.append(ed_mask)
        return torch.stack(ed_repr_list, dim=0), torch.stack(ed_mask_list, dim=0)

    def graph_info(self,context_index,es_len = None ,es_idx=None,ed_idx=None, graph=None,es_mask=None):

        his_src_mask = get_pad_mask(context_index, self.pad_idx)  # B, T (torch.bool) for attention mask
        enc_context_pooler = self.his_encoder(context_index, his_src_mask)['pooler_output']
        context_rep = self.conv_context_entity(enc_context_pooler)

        edge_idx, edge_type = graph
        topic_rep = self.RGCN(None, edge_idx, edge_type)

        conv_topic_rep = self.conv_RGCN(None, edge_idx, edge_type)

        [bs, es_num] = list(es_idx.size())
        [bs, ed_num] = list(ed_idx.size())
        size = context_rep.size()

        es_rep = torch.index_select(topic_rep, dim=0, index=es_idx.long().view(-1)).reshape(bs, es_num, -1)
        ed_rep = torch.index_select(topic_rep, dim=0, index=ed_idx.long().view(-1)).reshape(bs, ed_num, -1)

        conv_es_rep = torch.index_select(conv_topic_rep, dim=0, index=es_idx.long().view(-1)).reshape(bs, es_num, -1)
        conv_ed_rep = torch.index_select(conv_topic_rep, dim=0, index=ed_idx.long().view(-1)).reshape(bs, ed_num, -1)

        context_rep_ = context_rep.view(size[0], 1, 1, -1).repeat(1, ed_num, es_num, 1)

        ed_rep_ = repeat(ed_rep, 'i j k -> i j a k', a=es_num)
        es_rep_ = repeat(es_rep, 'i j k -> i a j k', a=ed_num)

        p_relation = self.prior(es_rep_, ed_rep_, context_rep_)

        hyp_relation = self.softmax(p_relation)

        hyp_relation = self.relation_gumbel(hyp_relation,training = False)

        bs, ed_num, es_num, relation_num = hyp_relation.size()
        es_mask = repeat(es_mask, 'i j -> i k j', k=ed_num)
        ed_prob = hyp_relation[:,:,:,-1].masked_fill(es_mask == 0, 0)
        ed_prob = torch.einsum('ikj->ik', ed_prob)
        es_mask = 1 / es_mask.sum(-1).float()
        ed_prob = torch.mul(ed_prob, es_mask)

        conv_ed_rep,ed_mask = self._encode_user_ed(ed_prob,es_len,conv_ed_rep,ed_idx)

        return conv_es_rep,conv_ed_rep,ed_mask

    def response_generation(self,context_index = None,es_idx = None, es_mask  = None,es_rep = None,
                                            ed_rep = None,ed_idx = None, ed_mask = None, resp_gth = None):

        context_mask = get_pad_mask(context_index, self.pad_idx).unsqueeze(-2)  # B, T (torch.bool) for attention mask
        enc_context_rep = self.tfr_encoder(context_index, context_mask)
        ed_rep = self.conv_entity_norm(ed_rep)
        es_rep = self.conv_entity_norm(es_rep)
        es_mask = es_mask.unsqueeze(-2)
        ed_mask = ed_mask.unsqueeze(-2)

        bs = context_index.size(0)
        if resp_gth is not None:
            resp_mask = get_pad_mask(resp_gth, self.pad_idx).unsqueeze(-2) & get_subsequent_mask(resp_gth)
            dec_out = self.decoder(resp_gth,resp_mask,enc_context_rep,context_mask,es_rep,es_mask,ed_rep,ed_mask)
            probs = self.proj(dec_out=dec_out,enc_context_rep=enc_context_rep,context_mask=context_mask,context_idx = context_index,
                              es_rep=es_rep,es_mask=es_mask,es_idx = es_idx,
                              ed_rep=ed_rep,ed_mask=ed_mask,ed_idx = ed_idx)
            return probs
        else:
            seq_gen = torch.ones(bs, 1, dtype=torch.long) * self.bos_idx
            seq_gen = seq_gen.cuda()
            if TO.beam_width == 1:
                seq_gen,probs = self._greedy_search(seq_gen=seq_gen, src_hidden=enc_context_rep, src_mask=context_mask,context_idx = context_index,
                                                      es_rep=es_rep,es_mask=es_mask,es_idx = es_idx,
                                                      ed_rep=ed_rep,ed_mask=ed_mask,ed_idx = ed_idx)
            else:
                seq_gen = self._beam_search(seq_gen=seq_gen, src_hidden=enc_context_rep, src_mask=context_mask,context_idx = context_index,
                                              es_rep=es_rep,es_mask=es_mask,es_idx = es_idx,
                                              ed_rep=ed_rep,ed_mask=ed_mask,ed_idx = ed_idx)
                probs = None
            return seq_gen,probs

    def proj(self, dec_out,enc_context_rep=None,context_mask=None,context_idx = None,
                              es_rep=None,es_mask=None,es_idx = None,
                              ed_rep=None,ed_mask=None,ed_idx = None ):
        B = dec_out.size(0)
        gen_logit = self.gen_proj(dec_out)
        L_r = dec_out.size(1)

        copy_logit_con = torch.bmm(dec_out, enc_context_rep.permute(0, 2, 1))
        copy_logit_con = copy_logit_con.masked_fill((context_mask == 0).expand(-1, L_r, -1), -1e9)

        copy_logit_es = torch.bmm(dec_out, es_rep.permute(0, 2, 1))
        copy_logit_es = copy_logit_es.masked_fill((es_mask == 0).expand(-1, L_r, -1), -1e9)

        copy_logit_ed = torch.bmm(dec_out, ed_rep.permute(0, 2, 1))
        copy_logit_ed = copy_logit_ed.masked_fill((ed_mask == 0).expand(-1, L_r, -1), -1e9)

        logits = torch.cat([gen_logit, copy_logit_con,copy_logit_es,copy_logit_ed], -1)
        if TO.scale_prj:
            logits *= self.hidden_size ** -0.5
        probs = torch.softmax(logits, -1)

        gen_prob = probs[:, :, :self.n_vocab]

        copy_context_prob = probs[:, :, self.n_vocab:self.n_vocab + enc_context_rep.size(1)]
        context = self.one_hot_scatter(context_idx, self.n_vocab)
        copy_context_prob = torch.bmm(copy_context_prob, context)

        copy_es_prob = probs[:, :,
                       self.n_vocab + enc_context_rep.size(1):self.n_vocab + enc_context_rep.size(1) + es_rep.size(1)]
        transfer_es_word = torch.gather(self.loc2glo.unsqueeze(0).expand(B, -1), 1, es_idx)
        copy_es_temp = copy_es_prob.new_zeros(B, L_r, self.n_vocab)
        copy_es_prob = copy_es_temp.scatter_add(dim=2,
                                                index=transfer_es_word.unsqueeze(1).expand(-1, L_r, -1),
                                                src=copy_es_prob)

        copy_ed_prob = probs[:, :, self.n_vocab + enc_context_rep.size(1) + es_rep.size(1):]
        transfer_ed_word = torch.gather(self.loc2glo.unsqueeze(0).expand(B, -1), 1, ed_idx)
        copy_ed_temp = copy_ed_prob.new_zeros(B, L_r, self.n_vocab)
        copy_ed_prob = copy_ed_temp.scatter_add(dim=2,
                                                index=transfer_ed_word.unsqueeze(1).expand(-1, L_r, -1),
                                                src=copy_ed_prob)

        probs = gen_prob + copy_context_prob + copy_es_prob + copy_ed_prob

        return probs


    def _greedy_search(self, seq_gen, src_hidden, src_mask,context_idx = None,
                              es_rep = None,es_mask = None,es_idx = None,
                              ed_rep = None,ed_mask = None,ed_idx = None):
        probs = None
        for step in range(DO.target_max_len_gen):
            single_step_probs = self.single_decode(input_seq=seq_gen, src_hidden=src_hidden, src_mask=src_mask,context_idx = context_idx,
                                                  es_rep = es_rep,es_mask = es_mask,es_idx = es_idx,
                                                  ed_rep = ed_rep,ed_mask = ed_mask,ed_idx = ed_idx )# B, 1, V
            if probs is None:
                probs = single_step_probs
            else:
                probs = torch.cat([probs, single_step_probs], 1)
            single_step_probs = single_step_probs.squeeze(1)
            single_step_word = single_step_probs.argmax(-1).unsqueeze(-1)
            seq_gen = torch.cat([seq_gen, single_step_word], 1)
        return seq_gen[:, 1:], probs

    def _single_decode(self, input_seq, src_hiddens, src_mask, input_mask=None,context_idx = None,
                              es_rep = None,es_mask = None,es_idx = None,
                              ed_rep = None,ed_mask = None,ed_idx = None,
                       ret_last_step=True):
        batch_size = input_seq.size(0)
        trg_seq_mask = get_subsequent_mask(input_seq)
        trg_seq_mask = trg_seq_mask.expand(batch_size, -1, -1)
        if input_mask is not None:
            trg_seq_mask = input_mask & trg_seq_mask
        dec_output = self.decoder(input_seq,trg_seq_mask,src_hiddens,src_mask,es_rep,es_mask,ed_rep,ed_mask)
        if ret_last_step:
            last_step_dec_output = dec_output[:, -1, :].unsqueeze(1)
            return last_step_dec_output
        else:
            return dec_output

    def single_decode(self, input_seq, src_hidden, src_mask, context_idx = None,
                              es_rep = None,es_mask = None,es_idx = None,
                              ed_rep = None,ed_mask = None,ed_idx = None):
        dec_output = self._single_decode(input_seq.detach(), src_hidden, src_mask, context_idx = context_idx,
                                        es_rep = es_rep, es_mask = es_mask, es_idx = es_idx,
                                        ed_rep = ed_rep, ed_mask = ed_mask, ed_idx = ed_idx)
        single_step_probs = self.proj(dec_out = dec_output, enc_context_rep = src_hidden, context_mask = src_mask, context_idx = context_idx,
                                        es_rep = es_rep, es_mask = es_mask, es_idx = es_idx,
                                        ed_rep = ed_rep, ed_mask = ed_mask, ed_idx = ed_idx)

        return single_step_probs


    def one_hot_scatter(self,indice, num_classes, dtype=torch.float):
        indice_shape = list(indice.shape)
        placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
        v = 1 if dtype == torch.long else 1.0
        placeholder.scatter_(-1, indice.unsqueeze(-1), v)
        return placeholder


    def forward(self,graph, inputs=None, pretrain=False):
        [resp,context_gen,context_index, _, _, es_idx, ed_idx,es_idx_len,es_mask] = inputs

        es_rep,ed_rep,ed_mask = self.graph_info(context_index,es_idx_len ,es_idx,ed_idx, graph,es_mask)

        if self.training:
            resp = self.response_generation(context_index=context_gen,es_idx=es_idx, es_mask = es_mask,es_rep = es_rep,
                                            ed_rep=ed_rep,ed_idx = ed_idx, ed_mask = ed_mask, resp_gth=resp)
            return resp
        else:
            resp, probs = self.response_generation( context_index=context_gen,es_idx=es_idx, es_mask = es_mask,es_rep = es_rep,
                                            ed_rep=ed_rep,ed_idx = ed_idx, ed_mask = ed_mask, resp_gth=None)
            return resp, probs


if __name__ == '__main__':
    total=torch.ones((2571,2571),dtype=torch.long).cuda()
    total_=torch.ones((2571,2571,5),dtype=torch.long).cuda()
    total__=torch.ones((16,20,200,44),dtype=torch.long).cuda
