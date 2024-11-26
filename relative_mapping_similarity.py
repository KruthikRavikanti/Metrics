import pandas as pd
import numpy as np
import editdistance
import hydra
from omegaconf import DictConfig
from scipy import stats
import ipdb

import math


def normalized_ed(s1, s2, method='max', epsilon=1e-10):
    """
    normalized edit distance between s1 and s2.
    parameters:
    s1 and s2: two strings
method:     The denominator is defined by method parameter.
    Choices:
    - max: the maximum of lengths of the two input strings
    - sum: the sum of lengths of two input strings
    - nes: 1-the normalized edit similarity to get distance.
    epsilon: small number to avoid dividing by zero in NES.
    default: 1e-10
    """
    # source: https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
    supported_methods = ['max', 'sum', 'nes']
    method = method.lower()
    if method not in supported_methods:
        raise Exception(f"unsupported normalization method {method}. Please choose from {supported_methods}")

    if len(s1) == 0 or len(s2) == 0:
        return 1
    d = editdistance.eval(s1, s2)
    if method == 'max':
        d = d / max(len(s1), len(s2))
    elif method == 'sum':
        d = d / (len(s1) + len(s2))
    elif method == 'nes':
        d = 1 - 1.0 / math.exp(d/(min(len(s1), len(s2)) - d + epsilon))
    return d

def compute_rms(
        t_pred: pd.DataFrame,
        t_true: pd.DataFrame,
        epsilon=1e-10,
        ):
    """
    Compute Relative Mapping Similarity (RMS) score.
    For further details about the algorithm, check
    https://arxiv.org/abs/2212.10505

    parameters:
    t_pred (pd.DataFrame): predicted table
    t_true (pd.DataFrame): ground truth table
    epsilon: small value used to avoid dividing by zero

    return:
    rms_precision: precision score
    rms_recall: recall score
    rms_f1: f1 score (harmonic mean between precision and recall)
    """
    # assume that index is ordered numbers that are not included in the generated tables.
    # thus, the first column is row headers.

    # the norma,normalized edit distance is calculated between 
    # :math: p^r || p^c and t^R || t^c
    # where || is concatenation operator
    pred_dict = {}
    for row in t_pred.itertuples():
        row_header = row[1]
        columns = [str(row_header) + '-' + col for col in t_pred.columns[1:]]
        pred_dict.update(dict(
            zip(columns, row[2:])
            ))
    true_dict = {}
    for row in t_true.itertuples():
        row_header = row[1]
        columns = [str(row_header) + '-' + col for col in t_true.columns[1:]]
        true_dict.update(dict(
            zip(columns, row[2:])
            ))

    # compute normalized edit distances
    #norm_eds, d_thetas, d_tau_thetas = {}, {}, {}
    #for k_pred, v_pred in pred_dict.items():
    #    norm_eds[k_pred] = {
    #            k_true: normalized_ed(k_pred, k_true)
    #            for k_true in true_dict.keys()
    #            }
    #    d_thetas[k_pred] = {
    #        k_true: abs(v_pred - v_true)/v_true
    #        for k_true, v_true in true_dict.items()
    #        }
    #    d_tau_theta = {}
    #    for k_true, ned, d_theta in zip(true_dict.keys(), norm_eds[k_pred].values(), d_thetas[k_pred].values()):
    #        d_tau_theta[k_true] = ned * d_theta
    #    d_tau_thetas[k_pred] = d_tau_theta.copy()
    ##  compute pairwise similarity matrix between keys
    similarity_matrix = {}
    binarized_similarity_matrix = {}
    for k_pred in pred_dict.keys():
        similarity_matrix[k_pred] = {
                k_true: 1-normalized_ed(k_pred, k_true)
                for k_true in true_dict.keys()
                }
        max_index = np.argmax(list(similarity_matrix[k_pred].values()))
        binarized_similarity_matrix[k_pred] = {
                k: (1 if i == max_index else 0)
                for i, k in enumerate(similarity_matrix[k_pred].keys())
                }

    # compute RMS f1
    numerator = 0
    for tuple_pred in pred_dict.items():
        for tuple_true in true_dict.items():
            norm_ed = normalized_ed(tuple_pred[0], tuple_true[0])
            d_theta = min(1, abs(tuple_pred[1] - tuple_true[1])/abs(max(tuple_true[1], epsilon)))
            d_tau_theta = (1 - norm_ed) * (1 - d_theta)
            numerator += binarized_similarity_matrix[tuple_pred[0]][tuple_true[0]] * d_tau_theta

    #ipdb.set_trace()
    rms_precision = numerator / len(pred_dict)
    rms_recall = numerator / len(true_dict)
    if rms_precision == 0 and rms_recall == 0:
        rms_f1 = 0
    else:
        rms_f1 = 2 * (rms_precision * rms_recall) / (rms_precision + rms_recall)

    return {
            'rms_precision': rms_precision,
            'rms_recall': rms_recall,
            'rms_f1': rms_f1,
            }

@hydra.main(config_path='.', config_name='rms', version_base='1.3.2')
def main(cfg: DictConfig):
    pass
if __name__ == '__main__':
    main()
