# -*- coding: utf-8 -*-

import scipy.stats as stats
from collections import OrderedDict
from tqdm import tqdm

def t_test(a_score_dict: OrderedDict, b_score_dict: OrderedDict):
    """t-test

    Args:
        a_score_dict:   OrderedDict instance which's key is string, value is np.ndarray
        b_score_dict:   ~
    """
    metric_names = list(a_score_dict.keys())
    efficient_metric_names = [x for x in metric_names if isinstance(a_score_dict[x], list)]
    a_scores = [a_score_dict[name] for name in efficient_metric_names]
    b_scores = [b_score_dict[name] for name in efficient_metric_names]
    p_values = [float(stats.ttest_ind(a, b).pvalue) for (a, b) in tqdm(zip(a_scores, b_scores))]
    name2p = dict(zip(efficient_metric_names, p_values))
    return dict([(name, name2p[name]) if name in name2p else (name, float("nan")) for name in metric_names])