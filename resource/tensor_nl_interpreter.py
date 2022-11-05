# -*- coding: utf-8 -*-


import torch
from resource.option.dataset_option import DatasetOption
import json

class TensorNLInterpreter:
    """Tensor NL Interpreter"""
    def __init__(self, vocab):
        self.vocab = vocab
        self.mid2name = json.load(open(DatasetOption.mid2name)) #for REDIAL dataset

    def interpret_tensor2nl(self, tensor,context = False):
        """interpret tensor to natural language
        Args:
            tensor(torch.Tensor): B, T

        Return:
            words(List(List(str))): B, T
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        sents=[]
        tensor=tensor.tolist()
        for sent in tensor:
            words = []
            for word in sent:
                word = self.vocab[word] # B, T
                # if '@' in word:
                #     movie_id = word[1:]
                #     if movie_id in self.mid2name:
                #         movie = self.mid2name[movie_id]
                #         tokens = movie.split(' ')
                #         words.extend(tokens)
                # else:
                #     words.append(word)
                words.append(word)
            sents.append(words)
        temp = []
        if context:
            for sent in sents:
                while "__pad__" in sent:
                    sent.remove('__pad__')
                sent.remove('__start__')
                temp.append(" ".join(sent))
            return temp

        for sent in sents:
                eos_index = len(sent)
                sos_index = 0
                try:
                    eos_index = sent.index("__end__")
                except (Exception, ):
                    pass
                try:
                    sos_index = sent.index("__start__")+1
                except (Exception, ):
                    pass

                sent = sent[sos_index:eos_index]
                temp.append(" ".join(sent))

        words = temp
        return words

    @staticmethod
    def word2sentence(words):
        sentences = [" ".join(word) for word in words]
        return sentences
