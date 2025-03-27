import pandas as pd
import numpy as np
import editdistance
import hydra
from omegaconf import DictConfig
from scipy import stats
import ipdb
from tqdm import tqdm
from typing import Iterable, List, Tuple, Dict

from pathlib import Path
import json
import math
import logging


logger = logging.getLogger()

def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.readlines()

def read_json_file(file):
    with open(file) as f:
        tmp = json.load(f)
    return tmp

def write_json_file(dictionary, file):
    with open(file, 'w') as f:
        json.dump(dictionary, f)

def read_markdown_as_dataframe(file):
    return pd.read_table(
            file, sep='|', skipinitialspace=True, header=0,
            ).dropna(axis=1).iloc[1:]

def find_files_list(dir):
    """
    for the given directory,
    find all predictions files with their corresponding types.
    Note: It assumes that all files are of the same type,
    e.g. all in json, or csv or markdown.

    supported extensions are:
    * json
    * csv
    * md
    """
    dir = Path(dir)
    supported_types = ['json', 'md', 'csv']
    files = {}
    for supported_type in supported_types:
        files[supported_type] = list(dir.rglob(f"*.{supported_type}"))
    _type = [k for k, v in files.items() if v][0]
    _files = [v for v in files.values() if v][0]
    return _files, _type
def read_table(file, _type):
    """
    given a file and its type,
    read the file accordingly
    """
    if _type == 'csv':
        df = pd.read_csv(file)
    elif _type == 'json':
        df = pd.read_json(file)
    elif _type == 'md':
        df = read_markdown_as_dataframe(file)
    else:
        raise Exception(f"unsupported file type {_type}")
    return df

def convert_dict_to_float_in_table(df):
    """
    Some tables contain dictionary in cells instead of float.
    I wrote this function to reformat those tables into more suitable form for computing RMS.
    """
    table = {}
    # create new column for first key, vvalue in the dictionary
    col_name = list(df.iloc[0, 0].keys())[0]
    table[col_name] = [dictionary[col_name] for dictionary in df.iloc[:, 0]]
    # extract the value from second key and insert it as float
    # in the cell instead of the dictionary
    for col in df.columns:
        table[col] = [list(dictionary.values())[-1] for dictionary in df[col]]

    return pd.DataFrame.from_dict(table)

def convert_tuple_to_float_in_table(df):
    """
    Some tables contain tuples or lists in cells instead of float.
    I wrote this function to reformat those tables into more suitable form for computing RMS.
    Note: the missing column name is assumed to be "index".
    TODO: You may need a smarter solution.
    """
    table = {}
    # create new column for first key, vvalue in the dictionary
    col_name = "index" # df.iloc[0, 0][0]
    table[col_name] = [_tuple[0] for _tuple in df.iloc[:, 0]]
    # extract the value from second key and insert it as float
    # in the cell instead of the tuple
    for col in df.columns:
        table[col] = [_tuple[-1] for _tuple in df[col]]

    return pd.DataFrame.from_dict(table)


def convert_list_to_float_in_table(*args, **kwargs):
    """
    alias to convert_tuple_to_float_in_table(df) since same handling algorithm.
    """
    return convert_tuple_to_float_in_table(*args, **kwargs)

def correct_format(df):
    """
    Function to include the logic for correcting any issues in the generated tables.
    """
    if isinstance(df.iloc[0,0], dict):
        df = convert_dict_to_float_in_table(df)
    elif isinstance(df.iloc[0,0], list) or isinstance(df.iloc[0,0], tuple):
        df = convert_tuple_to_float_in_table(df)

    return df

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
            # we found some tables to include tuples as cell values,
            # e.g. [2007, 54]
            if isinstance(tuple_pred[1], float) or isinstance(tuple_pred[1], int):
                d_theta = min(1, abs(tuple_pred[1] - tuple_true[1])/abs(max(tuple_true[1], epsilon)))
            elif isinstance(tuple_pred[1], str):
                tuple_pred = (tuple_pred[0], float(tuple_pred[1].strip()))
                d_theta = min(1, abs(tuple_pred[1] - tuple_true[1])/abs(max(tuple_true[1], epsilon)))
            elif isinstance(tuple_pred[1], Tuple) or isinstance(tuple_pred[1], List):
                d_theta = min(1, abs(tuple_pred[1][1] - tuple_true[1])/abs(max(tuple_true[1], epsilon)))
            elif isinstance(tuple_pred[1], Dict):
                d_theta = min(1, abs(list(tuple_pred[1].values())[1] - tuple_true[1])/abs(max(tuple_true[1], epsilon)))
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

@hydra.main(config_path='conf', config_name='rms', version_base='1.3.2')
def main(cfg: DictConfig):
    setattr(cfg, 'true_dir', Path(cfg.true_dir))
    setattr(cfg, 'pred_dir', Path(cfg.pred_dir))
    true_files, true_files_type = find_files_list(cfg.true_dir)
    pred_files, pred_files_type = find_files_list(cfg.pred_dir)
    scores = []
    true_files = sorted(true_files)
    pred_files = sorted(pred_files)
    if len(true_files) != len(pred_files):
        if cfg.skip_non_existing_tables:
            num_removed = len(true_files) - len(pred_files)
            pred_stems = {f.stem for f in pred_files}
            true_files = [f for f in true_files if f.stem in pred_stems]
            print(f"Warning: ignoring {num_removed} tables because there is no corresponding predictions.")
            print(f"computing score on {len(true_files)} tables")
        else:
            raise Exception("different lengthss of csv files list"
                            f"true files: {len(true_files)}, prediction files: {len(pred_files)}")

    # We assume that there are only two levels:
    # e.g. chart_type/image_id.csv
    fail_count = 0
    for true_file, pred_file in tqdm(zip(true_files, pred_files), total=len(true_files)):
        try:
            df_true = read_table(true_file, true_files_type)
            df_pred = read_table(pred_file, pred_files_type)
        except:
            fail_count += 1
            continue
        pred_content = "".join(read_file(pred_file))
        if (
                'python' in pred_content
            or 'print' in pred_content
            ):
            fail_count += 1
            continue
        try: # if len(df_pred) > 0:
            df_pred = correct_format(df_pred)
            scores.append({
                'pred_file': pred_file.parent / pred_file.stem,
                **compute_rms(df_pred, df_true),
                'is_success': True,
                })
        except: # else:
            fail_count += 1
            scores.append({
                'pred_file': pred_file.parent / pred_file.stem,
                'rms_precision': 0, 'rms_recall': 0, 'rms_f11': 0,
                'is_success': False,
                })

    df_scores = pd.DataFrame.from_dict(scores)
    overall_scores = {score: round(df_scores[score].mean(), 4) for score in df_scores.columns if score not in ['pred_file']}
    success_rate = df_scores['is_success'].value_counts()[True] / len(df_scores)
    logger.info(f"overall scores are: \n{overall_scores}",
                #[f"{score}: {round(df_scores[score].mean(), 4)}" for score in df_scores.columns if score not in ['pred_file']]
                 )
    #logger.info(f"failing count: {fail_count}")
    #logger.info(f"success rate: {success_rate}")
    logger.info(f"saving scores to {cfg.scores_csv}")
    setattr(cfg, 'scores_csv', Path(cfg.scores_csv))
    df_scores.to_csv(cfg.scores_csv)
    overall_csv = cfg.scores_csv.parent / f"{cfg.scores_csv.stem}.overall.csv"
    logger.info(f"saving overall scores to {overall_csv}")
    pd.DataFrame.from_dict([overall_scores]).to_csv(overall_csv)
if __name__ == '__main__':
    main()
