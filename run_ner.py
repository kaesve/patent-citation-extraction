from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from sklearn.metrics import classification_report

from time import time

import joblib
import json

from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import ann2bio


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, tokens_a=None, text_b=None, tokens_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: (Optional) string. The untokenized text of the first
                sequence. Either text_a or tokens_a need to be specified.
            tokens_a: (Optional) [string]. Words of the first sequence. Either
                 text_a or tokens_a need to be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            tokens_b: (Optional) string. The words of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid

        if text_a == None and tokens_a == None:
            raise ValueError("At least one of `text_a` and `tokens_a` must be specified.")

        self.text_a = text_a if text_a != None else " ".join(tokens_a)
        self.tokens_a = tokens_a if tokens_a != None else text_a.split(" ")


        if text_b != None or tokens_b == None:
            self.text_b = text_b
        else:
            self.text_b = " ".join(tokens_b)

        if tokens_b != None or text_b == None:
            self.tokens_b = tokens_b
        else:
            self.tokens_b = text_b.split(" ")

        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


def load_bio_file(filename, sep=' '):
    '''
    read .bio file and extract sequences of words and labels
    '''
    with open(filename, encoding="utf-8") as f:
        data = []
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            if line == "" or line.startswith('-DOCSTART'):
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
            else:
                splits = line.split(sep)
                sentence.append(splits[0])
                label.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label))
    return data

class NerProcessor(DataProcessor):
    """Processor for datasets using the BIO format."""

    def __init__(self, labels=["B", "I", "O" ]):
        self.labels = labels + [ "[CLS]", "[SEP]" ]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, tokens_a=sentence, label=label))
        return examples
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return load_bio_file(input_file)



class PatentProcessor(NerProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, train_ratio = 17/22):
        """See base class."""
        patent_files = os.listdir(data_dir)
        patent_train_files = patent_files[:round(len(patent_files)*train_ratio)]
        return self._create_examples(data_dir, patent_train_files, "train")

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(data_dir, patent_test_files, "dev")

    def get_test_examples(self, data_dir, train_ratio = 17/22):
        """See base class."""
        patent_files = os.listdir(data_dir)
        patent_test_files = patent_files[round(len(patent_files)*train_ratio):]
        return self._create_examples(data_dir, patent_test_files, "test")

    def get_all_examples(self, data_dir):
        """See base class."""
        patent_files = os.listdir(data_dir)
        return self._create_examples(data_dir, patent_files, "all")
    
    def _create_examples(self, data_dir, files, set_type):
        examples = []
        for f in files:
            if f[-4:] == ".bio":
                sentence, label = load_bio_file(os.path.join(data_dir, f), sep="\t")[0]
            elif f[-4:] == ".txt":
                with open(os.path.join(data_dir, f), "r", encoding="utf-8") as txt:
                    tokens = ann2bio.process_patent(txt.read())
                    sentence, _, label = zip(*tokens)
            
            examples.append(InputExample(guid=f, tokens_a=sentence, label=label))


        return examples

    def convert_examples_to_features(self, examples, label_map, max_seq, tokenizer):
        return convert_examples_to_features(examples, label_map, max_seq, tokenizer, overflow_mode="chunk")
    
    def convert_feature_pred_to_example_pred(self, predictions, examples, tokenization_splits):
        results = []
        i = 0
        for ex, chunks in zip(examples, tokenization_splits):
            current_thing = {
                "errors": [],
                "total_abs_error": 0,
                "total_error": 0,
                "chunks": chunks,
                "predictions": []
            }

            abs_total_err = 0
            total_err = 0
            for split in chunks:
                split_pred = predictions[i]

                error = len(split_pred) - split
                current_thing["errors"].append(error)
                current_thing["total_abs_error"] += abs(error)
                current_thing["total_error"] += error
                current_thing["predictions"].extend(split_pred)

                i += 1
            
            results.append(current_thing)

        assert len(predictions) == i

        return results

def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, overflow_mode="clip", split_overlap=0):
    """Loads a data file into a list of `InputBatch`s."""

    result_input_ids = []
    result_input_mask = []
    result_segment_ids = []
    result_label_id = []
    result_valid_ids = []
    result_label_mask = []

    result_tokenization_splits = []

    max_tokens = max_seq_length - 2
    def pad(ls, v):
        return ls + [v] * (max_seq_length - len(ls))

    for (ex_index,example) in enumerate(examples):
        textlist = example.tokens_a
        labels = example.label
        assert len(textlist) == len(labels)

        tokens_chunked = []
        valid_chunked = []
        labels_chunked = []
        current_splits = []

        current_start = 0
        current_tokens = []
        current_valid = []
        i = 0
        for word in textlist:
            token = tokenizer.tokenize(word)
            assert len(token) > 0

            if len(token) + len(current_tokens) > max_tokens:

                # in rare cases, len(token) > max_tokens (when we have a very long word)
                # in that case, current_tokens can be longer than max_tokens, so we slice
                tokens_chunked.append(current_tokens[:max_tokens])
                valid_chunked.append(current_valid[:max_tokens])
                labels_chunked.append(labels[current_start:i])
                current_splits.append(i - current_start)

                current_start = i
                current_tokens = []
                current_valid = []
                

            current_tokens.extend(token)
            current_valid.extend([1] + [0] * (len(token) - 1))

            i += 1

        if len(current_tokens) > 0:
            # in rare cases, len(token) > max_tokens (when we have a very long word)
            # in that case, current_tokens can be longer than max_tokens, so we slice
            tokens_chunked.append(current_tokens[:max_tokens])
            valid_chunked.append(current_valid[:max_tokens])
            labels_chunked.append(labels[current_start:i])
            current_splits.append(i - current_start)

        result_tokenization_splits.append(current_splits)

        if overflow_mode == "clip":
            chunk_count = 1
        else:
            chunk_count = len(tokens_chunked)

        for i in range(chunk_count):

            tokens = [ "[CLS]", *(tokens_chunked[i]), "[SEP]" ]
            labels = [ "[CLS]", *(labels_chunked[i]), "[SEP]" ]
            valid = pad([ 1, *(valid_chunked[i]), 1 ], 1)

            input_mask = pad([1] * len(tokens), 0)
            label_mask = pad([1] * len(labels), 0)
            segment_ids = [0]*max_seq_length

            label_ids = pad([ label_map[l] for l in labels ], 0)
            input_ids = pad(tokenizer.convert_tokens_to_ids(tokens), 0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length


            result_input_ids.append(input_ids)
            result_input_mask.append(input_mask)
            result_segment_ids.append(segment_ids)
            result_label_id.append(label_ids)
            result_valid_ids.append(valid)
            result_label_mask.append(label_mask)

    all_input_ids = torch.tensor(result_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(result_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(result_segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(result_label_id, dtype=torch.long)
    all_valid_ids = torch.tensor(result_valid_ids, dtype=torch.long)
    all_lmask_ids = torch.tensor(result_label_mask, dtype=torch.long)

    result = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
    return result, result_tokenization_splits

def prep_model(model, optimizer, n_gpu, args):
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)
    
    return model, optimizer

def train_on_examples(examples, args, processor, label_map, config, tokenizer, device, n_gpu):
    train_data, tokenization_splits = processor.convert_examples_to_features(examples, label_map, args.max_seq_length, tokenizer)

    batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



    model = Ner.from_pretrained(args.bert_model, from_tf=False, config=config)
    model.to(device)


    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    
    num_train_optimization_steps = int(len(train_data.tensors[0]) / batch_size / args.gradient_accumulation_steps)
    num_train_optimization_steps *= args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)


    model, optimizer = prep_model(model, optimizer, n_gpu, args)
    


    logger.info("***** Running training *****")
    logger.info("  Num sentences = %d", len(examples))
    logger.info("  Num examples = %d", len(train_data.tensors[0]))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
    return model

def evaluate_on_examples(eval_examples, args, processor, label_map, label_list, model, tokenizer, device, n_gpu):
    eval_data, tokenization_splits = processor.convert_examples_to_features(eval_examples, label_map, args.max_seq_length, tokenizer)
    
    logger.info("***** Running evaluation *****")
    logger.info("  Num sentences = %d", len(eval_examples))
    logger.info("  Num examples = %d", len(eval_data.tensors[0]))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    y_true = []
    y_pred = []
    for data in tqdm(eval_dataloader, desc="Evaluating"):
        data = tuple(t.to(device) for t in data)
        input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask = data

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, labels in enumerate(label_ids):
            for j, l_id in enumerate(labels):
                if l_id == label_map["[SEP]"]:
                    y_true.append([ label_list[l - 1] for l in label_ids[i][1:j] ])
                    y_pred.append([ label_list[l - 1] for l in logits[i][1:j] ])
                    break

    y_pred = processor.convert_feature_pred_to_example_pred(y_pred, eval_examples, tokenization_splits)
    return y_pred

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--do_leave_one_out",
                        action='store_true',
                        help="Whether to run leave one out evaluation.")
    parser.add_argument("--train_ratio",
                        default=0.77,
                        type=float,
                        help="Proportion of the dataset used for training vs evaluation")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if not args.do_train and not args.do_eval and not args.do_leave_one_out:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_leave_one_out` must be True.")

    if args.do_leave_one_out and (args.do_train or args.do_eval):
        raise ValueError("`do_leave_one_out` should be run separately from `do_train` and `do_eval`.")

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = PatentProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1 # +1 to account for padding
    label_map = { label: i for i, label in enumerate(label_list,1)}


    if args.do_train or args.do_leave_one_out:

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        # Prepare model
        config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

        if args.gradient_accumulation_steps < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)


        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir, args.train_ratio)
            model = train_on_examples(train_examples, args, processor, label_map, config, tokenizer, device, n_gpu)

            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":num_labels,"label_map":label_map}
            json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        else:
            examples = processor.get_all_examples(args.data_dir, args.train_ratio)

            total_y_pred = []
            total_y_true = []
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            for left_out in examples:
                examples_without = [ ex for ex in examples if not ex is left_out ]
                model = train_on_examples(examples_without, args, processor, label_map, config, tokenizer, device, n_gpu)
            
                eval_results = evaluate_on_examples([ left_out ], args, processor, label_map, label_list, model, tokenizer, device, n_gpu)
                y_pred = [ res["predictions"] for res in eval_results ]
                y_true = [ left_out.label ]

                report = metrics.flat_classification_report(y_true, y_pred, labels=["B", "I"])
                logger.info("\n%s", report)
                with open(output_eval_file, "a+") as writer:
                    logger.info(f"***** Eval results for {left_out.guid} *****")
                    logger.info("\n%s", report)
                    writer.write(report)

                total_y_pred.extend(y_pred)
                total_y_true.extend(y_true)


            report = metrics.flat_classification_report(total_y_true, total_y_pred, labels=["B", "I"])
            logger.info("\n%s", report)
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Eval results total *****")
                logger.info("\n%s", report)
                writer.write(report)
            output_eval_file = os.path.join(args.output_dir, "eval_data.json")
            with open(output_eval_file, "w") as writer:
                json.dump(dict(zip([ ex.guid for ex in examples ], total_y_pred)), writer)

    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)

        model, optimizer = prep_model(model, None, n_gpu, args)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir, args.train_ratio)
        
        eval_results = evaluate_on_examples(eval_examples, args, processor, label_map, label_list, model, tokenizer, device, n_gpu)
        y_pred = [ res["predictions"] for res in eval_results ]
        y_true = [ ex.label for ex in eval_examples ]

        report = metrics.flat_classification_report(y_true, y_pred, labels=["B", "I"])

        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_report.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)

        patent_test_files = os.listdir(args.data_dir)[-5:]
        output_eval_file = os.path.join(args.output_dir, "eval_predictions.json")
        with open(output_eval_file, "w") as writer:
            json.dump(dict(zip(patent_test_files, y_pred)), writer)

        output_eval_file = os.path.join(args.output_dir, "eval_results.json")
        with open(output_eval_file, "w") as writer:
            json.dump(dict(zip(patent_test_files, eval_results)), writer)

        joblib.dump([logits, label_ids, input_mask, y_pred, y_true], os.path.join(args.output_dir, "output.bin"))
    with open(os.path.join(args.output_dir, f"run_{time()}.sh"), "w") as file:
        file.write("python " + " ".join(sys.argv))


if __name__ == "__main__":
    main()