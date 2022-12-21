import ipdb
import json
from resource.option.dataset_option import DatasetOption as DO
import jieba
from rouge import Rouge
def cal_calculate(outs):
    if DO.dataset == "Redial":
        mid2name = json.load(open(DO.mid2name))

    unigram_count = 0
    bigram_count = 0
    trigram_count = 0
    quagram_count = 0
    unigram_set = set()
    bigram_set = set()
    trigram_set = set()
    quagram_set = set()

    if DO.dataset == "Redial":
        for sentence in outs:
            sentence = sentence.split(" ")

            # full_sen_gen = []
            # for word in sentence:
            #     if '@' in word:
            #         movie_id = word[1:]
            #         if movie_id in mid2name:
            #             movie = mid2name[movie_id]
            #             tokens = movie.split(' ')
            #             full_sen_gen.extend(tokens)
            #         else:
            #             full_sen_gen.append(word)
            #     else:
            #         full_sen_gen.append(word)

            full_sen_gen = sentence

            for word in full_sen_gen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(full_sen_gen) - 1):
                bg = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(full_sen_gen) - 2):
                trg = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1]) + ' ' + str(full_sen_gen[start + 2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(len(full_sen_gen) - 3):
                quag = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1]) + ' ' + str(full_sen_gen[start + 2]) + ' ' + str(full_sen_gen[start + 3])
                quagram_count += 1
                quagram_set.add(quag)

    if DO.dataset == "TG":
        for sen in outs:
            sen_split_by_movie = list(sen.split('__unk__'))
            sen_1 = []
            for i, sen_split in enumerate(sen_split_by_movie):
                segment = sen_split.split(" ")
                sen_1.extend(segment)
                if i != len(sen_split_by_movie) - 1:
                    sen_1.append('__unk__')

            full_sen_gen = sen_1

            for word in full_sen_gen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(full_sen_gen) - 1):
                bg = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(full_sen_gen) - 2):
                trg = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1]) + ' ' + str(full_sen_gen[start + 2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(len(full_sen_gen) - 3):
                quag = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1]) + ' ' + str(
                    full_sen_gen[start + 2]) + ' ' + str(full_sen_gen[start + 3])
                quagram_count += 1
                quagram_set.add(quag)

    dis1 = len(unigram_set) / unigram_count  # unigram_count
    dis2 = len(bigram_set) / bigram_count  # bigram_count
    dis3 = len(trigram_set) / trigram_count  # trigram_count
    dis4 = len(quagram_set) / quagram_count  # quagram_count
    return dis1, dis2, dis3, dis4


def _cal_rouge(hypothesis,reference):
    """
    both hypothesis and reference are str
    """
    if hypothesis == '':
        return 0,0,0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-1']['f'],scores[0]['rouge-2']['f'],scores[0]['rouge-l']['f']
    except:
        print("something wrong here! ",hypothesis,reference)
        return 0,0,0
    

def cal_rouge(hypothesis_list,reference_list,dataset = "Redial",identities =None):
    """
    hypothesis_list & reference_list �?list of str
    """
    rouge1_sum, rouge2_sum, rouge_l_sum,cnt = 0,0,0,0
    for hypothesis,reference,identity in zip(hypothesis_list,reference_list,identities):
        rouge1,rouge2,rouge_l = _cal_rouge(hypothesis,reference)
        rouge1_sum += rouge1
        rouge2_sum += rouge2
        rouge_l_sum += rouge_l
        cnt += 1
    return rouge1_sum/cnt , rouge2_sum/cnt , rouge_l_sum/cnt