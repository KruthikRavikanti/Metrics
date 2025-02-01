import numpy as np
import pandas as pd
import regex
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import ipdb

from multiprocessing import Pool
import os
import json
import pickle
from pathlib import Path
import logging

from my_constants import *

logger = logging.getLogger("my_helping_functions")
logger.setLevel('DEBUG')


def is_sorted(arr):
    i = arr[0]
    for j in arr[1:]:
        if i>j:
            return False
        else:
            i=j
    return True

def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.readlines()

def read_files(files):
    tmp=[]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tmp.append(line)

    return tmp

def read_files_into_separate_lists(files):
    tmp=[]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            tmp.append(f.readlines())

    return tmp

def write_file(lines, file, create_parents=True, split_lines=True):
    file = Path(file)
    if create_parents:
        file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'w', encoding='utf-8') as f:
        if split_lines:
            f.writelines([line.strip() + '\n' for line in lines])
        else:
            f.write(lines)
    return None

def write_files(dict, create_parents=True):
    # keys are file names
    # values are list of lines to write
    for file, lines in dict.items():
        write_file(lines, file, create_parents=create_parents)
    return None

# similar functions but for json
def read_json_file(file):
    with open(file) as f:
        tmp = json.load(f)
    return tmp

def write_json_file(dictionary, file):
    with open(file, 'w') as f:
        json.dump(dictionary, f)

# similar functions but for pickle
def read_pickle_file(file):
    with open(file, 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def write_pickle_file(dictionary, file):
    with open(file, 'wb') as f:
        pickle.dump(dictionary, f)

def read_markdown_as_dataframe(file):
    return pd.read_table(
            file, sep='|', skipinitialspace=True, header=0,
            ).dropna(axis=1).iloc[1:]

# directories operations
def find_images(dir):
    """
    find list of images in the given directory.
    """
    images_extensions = ['.jpg', '.jpeg', '.png']
    #[yield p for p in Path(dir).rglob('*') if p.suffix.lower() in images_extensions]
    for p in Path(dir).rglob('*'):
        if p.suffix in images_extensions:
            yield p

# parallelism/speed related
def parallelize_list(l, n_cores=os.cpu_count()):        
    if n_cores<=0:
        n_cores=1
    parallelizable_data=[]
    if l:
        step_size=int(len(l)/n_cores)
        for i in range(0, len(l), step_size):
            parallelizable_data.append(l[i:i+step_size]) if i+step_size < len(l) else parallelizable_data.append(l[i:])

    return parallelizable_data

def parallelize_np_array(l, n_cores=os.cpu_count()):
    parallelizable_data=[]
    if len(l)>0:
        step_size=len(l)//n_cores
        for i in range(0, len(l), step_size):
            parallelizable_data.append(l[i:i+step_size]) if i+step_size < len(l) else parallelizable_data.append(l[i:])

    return parallelizable_data




def serialize_list(l):
    # Note: it serializes only one level.
    # for example, 
    # l=[[1, 2], [3, 4]]
    # will be l=[1, 2, 3, 4]
    serial_l=[]
    for sublist in l:
        for line in sublist:
            serial_l.append(line)

    return serial_l

def convert_into_dictionary(keys, values):
    dict={}
    for k, v in zip(keys, values):
        dict[k]=v

    return dict
def get_extension(file, char_before_extension=DOT):
    return file[file.rfind(char_before_extension)+1:]

def get_main_data_files(dir, src, tgt):
    allowed_files=(TRAIN, VALID, TEST)
    allowed_extensions=(src, tgt)
    files=[]
    for file in os.listdir(dir):
        tmp=file.split(DOT)
        if tmp[0] in allowed_files and tmp[1] in allowed_extensions:
            files.append(dir+file)
    return files

def get_splitted_data_directories(dir, src, tgt):
    allowed_files=(TRAIN, VALID, TEST)
    allowed_extensions=(src, tgt)
    files=[]
    for file in os.listdir(dir):
        tmp=file.split('_')
        if tmp[0] in allowed_files and tmp[1] in allowed_extensions:
            files.append(dir+file)
    return files

def get_segmented_files_in(dir):
    tmp=[]
    for file in os.listdir(dir):
        if '-segmented' in file:
            tmp.append(dir+file)

    return tmp

def get_postsegmented_files_in(dir):
    tmp=[]
    for file in os.listdir(dir):
        if '-postsegmented' in file:
            tmp.append(dir+file)

    return tmp

def convert_name_into_segmented_name(file):
    lang=get_extension(file)
    return file+'-segmented.'+lang

def convert_name_into_postsegmented_name(file):
    lang=get_extension(file)
    if '-segmented' in file:
        file=file[:file.rfind('-segmented')]
    return file+'-postsegmented.'+lang

def convert_name_into_bped_name(file):
    lang=get_extension(file)
    file=file[:file.rfind('-')+1]
    return file+'bpe.'+lang





def wait_till_finish_writing(file, reference_file=None):
    from subprocess import Popen, PIPE
    import shlex
    from time import sleep
    if not reference_file:
        reference_file=file[:file.rfind('-')]
    reference_num_lines=None
    while not reference_num_lines:
        reference_num_lines=Popen(shlex.split('wc -l '+reference_file), stdout=PIPE, stderr=PIPE).communicate()[0].decode('ascii')
    reference_num_lines=int(reference_num_lines.split()[0])
    num_lines=0
    while num_lines!=reference_num_lines:
        cmd_out=None
        while not cmd_out:
            cmd_out=Popen(shlex.split('wc -l '+file), stdout=PIPE, stderr=PIPE).communicate()[0].decode('ascii')
        if 'No such file or directory' not in cmd_out:
            num_lines=int(cmd_out.split()[0])
        sleep(1)

    return True


def string_to_nums(column):
    """
    input:
    column: pd.Series: contains all categories in string format.
    
    output:
    column: pd.Series: containing numbers instead of strings in the original column.
    mapping: dict: mapping strings to numbers.
    """
    mapping_dict = {string: index for index, string in enumerate(column.unique())}
    new_column = column.replace(mapping_dict)
    return (new_column, mapping_dict)
def expand_columns(df, column_to_expand, column_of_values, default='Neutral'):
    for category in df[column_to_expand].unique():
        df[category] = df.apply(lambda x: x[column_of_values] if x[column_to_expand] == category else default, axis='columns')

def convert_string_to_words_list(string):
    pattern = regex.compile(r"\w+")
    return pattern.findall(string)

# torch related functionalitieds
def get_device(device=None):
    return (device
            or ('cuda' if torch.cuda.is_available() else None)
            or ('mps' if torch.backends.mps.is_available() else None)
            or 'cpu'
            )


def cosine_similarity_2d(tensor1, tensor2):
    """
    for given 2x2-D tensors,
    this function computes cosine similarity between all elements in both tensors.
    I need that since in a lot of codes, so I wrote it here.
    """
    norm_t1 = tensor1 / np.linalg.norm(tensor1, axis=-1, keepdims=True)
    norm_t2 = tensor2 / np.linalg.norm(tensor2, axis=-1, keepdims=True)
    return norm_t1 @ norm_t2.T


class TorchDatasetWithHFTokenizer(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, inference_device='cuda', max_len=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.tokenizer = tokenizer
        self.inference_device = inference_device
        self.max_len = max_len
        self.sequences_longer_than_tokenizer_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def to(self, batch, device):
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def collate_fn(self, batch):
        try:
            tokenized = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    padding='longest', pad_to_multiple_of=8,
                    )
        except:
            self.sequences_longer_than_tokenizer_count += 1
            tokenized = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    padding='longest', pad_to_multiple_of=8,
                    )
        if tokenized['input_ids'].size(1) > self.max_len:
            self.sequences_longer_than_tokenizer_count += 1
            tokenized = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    padding='longest', pad_to_multiple_of=8,
                    )
        return self.to(tokenized, self.inference_device)


class BatchedTorchDatasetWithHFTokenizer(torch.utils.data.Dataset):
    """
    This class tokenizes the data at the while creating this class and saves the tokenized version for next loads.
/bin
This is necessary when we want to use max-tokens instead of batch-size.
Example:
    sentences = BatchedTorchDatasetWithHFTokenizer('/Users/mohamedfayed/data/mt/all.en', tokenizer, inference_device='cpu', max_tokens_per_gpu=4096, count_paddings=True, sort_according_to_length=True, save_batches=True)
    loader = torch.utils.data.DataLoader(sentences, collate_fn=sentences.collate_fn, batch_size=1, shuffle=False, multiprocessing_context=None)
    """
    def __init__(
            self,
            textfile,
            tokenizer,
            inference_device=None,
            max_tokens_per_gpu=None,
            bindir=None,
            sort_according_to_length=True,
            count_paddings=True,
            save_batches=False,
            *args, **kwargs
            ):
        """
        given datafile, this function creates preprocessed version and saves tokenized file in pyarrow format.
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len_single_sentence = self.tokenizer.max_len_single_sentence
        self.max_tokens_per_gpu = max_tokens_per_gpu if max_tokens_per_gpu else tokenizer.max_len_single_sentence
        self.inference_device = inference_device
        self.count_paddings = count_paddings
        self.sort_according_to_length = sort_according_to_length
        self.sequences_longer_than_tokenizer_count = 0
        self.batches_longer_than_tokenizer_count = 0
        logger.info(f"max tokens per gpu: {self.max_tokens_per_gpu}")
        if not bindir:
            bindir = str(textfile) + '.bin'
            self.bindir = bindir
            batchesfile = textfile + f'.batches.{max_tokens_per_gpu}'
            self.batchesfile = batchesfile
        self.data = self.load_data(textfile, bindir)
        if Path(batchesfile).exists():
            logger.info(f"loading batches from {batchesfile}")
            self.batches = read_pickle_file(batchesfile)
        else:
            logger.info("creating batches")
            self.batches = self.make_batches(self.data,
                                             self.max_tokens_per_gpu,
                                             count_paddings=count_paddings,
                                             sort_according_to_length=sort_according_to_length)
            if save_batches:
                write_pickle_file(self.batches, batchesfile)
        logger.info(f"data is converted into {len(self.batches)} batches")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def load_data(self, textfile, bindir=None, num_proc=min(48, os.cpu_count())):
        if Path(bindir).exists():
            logger.info(f"loading binary files from {bindir}")
            data = load_from_disk(bindir)
        else:
            logger.info(f"Reading text file {textfile}")
            data = load_dataset('text', data_files=[textfile])
            logger.info(f"reformatting data")
            data = data.map(lambda x, idx: {'text': x['text'], 'id': idx}, with_indices=True, batched=True, num_proc=num_proc)
            logger.info("tokenizing data")
            data = data.map(lambda x: {'tokens': self.tokenizer(x['text'], truncation=True, padding='longest', pad_to_multiple_of=8)['input_ids']}, num_proc=num_proc)
            data = data.map(lambda x: {'len': len(x['tokens'])}, num_proc=num_proc)
            logger.info(f"saving version including id and length arrow form to {bindir}")
            data.save_to_disk(bindir)

        return data

    def make_batches(
            self, data,
            max_tokens_per_gpu=2048,
            sort_according_to_length=True,
            count_paddings=True,
            num_proc=1,
            ):
        """
        convert the data into list of batches.
        If you want to train, make sure to shuffle data and rerun this function.
        count_paddings: bool=True:
        if True, it will consider padding to 'longest' in calculation of batch_size.
        num_processes: int = 1:
        If greater than 1, the function divides the data into num_processes segments equally,
        and generate batches, then merge all of them into single list again.
        This may be useful to speed up, but it will results into num_processes batches to be smaller than expected.
        This should not be a big deal in large datasets.
        """
        batches = []
        current_batch = []
        current_batch_tokens_count = 0
        max_tokenized_sequence_length_in_batch = 0
        if sort_according_to_length:
            # I sort descendingly to make it easier in translation task to determine which max tokens causes CUDA OOM
            # since larger sequences are most probably subject to larger generations
            logger.info("sorting dataset descendingly")
            data = data.sort('len', reverse=True)
            logger.info("done sorting, start batching")

        #ipdb.set_trace()
        for item in tqdm(data['train']):
            id = item['id']
            seq, tok_seq = item['text'], item['tokens']
            seq_len = len(tok_seq)
            max_tokenized_sequence_length_in_batch = max(len(tok_seq), max_tokenized_sequence_length_in_batch)
            if (
                    (count_paddings and (len(current_batch)+1) * max_tokenized_sequence_length_in_batch > max_tokens_per_gpu)
                    or (current_batch_tokens_count + seq_len > max_tokens_per_gpu)
                    ):
                if len(current_batch) > 0:
                    batches.append(current_batch)
                current_batch = [(id, seq, seq_len)]
                current_batch_tokens_count = seq_len
                max_tokenized_sequence_length_in_batch = seq_len
            else:
                current_batch.append((id, seq, seq_len))
                current_batch_tokens_count += seq_len
                #max_tokenized_sequence_length_in_batch = len(tok_seq)

        if len(current_batch) > 0:
            batches.append(current_batch)

        return batches

    def to(self, batch, device):
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def collate_fn(self, batch):
        assert len(batch) == 1, f"batch size should equal 1"
        batch = batch[0]
        ids = [item[0] for item in batch]
        sentences = [item[1] for item in batch]
        #ipdb.set_trace()
                #ipdb.set_trace())
        try:
            tokenized = self.tokenizer(
                    sentences,
                    return_tensors='pt',
                    padding='longest', pad_to_multiple_of=8,
                    )
        except Exception as e:
            self.sequences_longer_than_tokenizer_count += 1
            tokenized = self.tokenizer(
                    sentences,
                    return_tensors='pt',
                    truncation=True,
                    padding='longest', pad_to_multiple_of=8,
                    )
        if tokenized['input_ids'].size(1) > self.max_len_single_sentence:
            self.sequences_longer_than_tokenizer_count += 1
            tokenized = self.tokenizer(
                    sentences,
                    return_tensors='pt',
                    truncation=True,
                    padding='longest', pad_to_multiple_of=8,
                    )
        #return self.to(tokenized, self.inference_device)
        #ipdb.set_trace()
        return {'ids': ids, 'input_ids': tokenized['input_ids'].to(self.inference_device, non_blocking=True)}


def make_loader_of_text_data(
        iterable, tokenizer, inference_device=get_device(),
        *args, **kwargs
        ):
    """
    I made this function to make it easy to iterate over text datasets.
    Especially because iterating over text and tokenizing it to pass it to a model isnot easily integrable in code.
    Note: all *args and **kwargs are passed to DataLoader and not Dataset.
    """
    dataset = TorchDatasetWithHFTokenizer(iterable, tokenizer, inference_device)
    loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            *args, **kwargs
            )
    return dataset, loader
