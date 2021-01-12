# Extract references from patents

Use BERT-based models to extract references to scientific literature in patent texts. This repository includes the preprocessing pipeline, the script to train and evaluate the BERT model and the data used to finetune the models.

Read more at [Improving reference mining in patents with BERT](https://arxiv.org/abs/2101.01039).

# Finetuned models

I finetuned three models for this project:

- [Based on BERT](https://huggingface.co/kaesve/BERT_patent_reference_extraction)
- [Based on BioBERT](https://huggingface.co/kaesve/BioBERT_patent_reference_extraction)
- [Based on SciBERT](https://huggingface.co/kaesve/SciBERT_patent_reference_extraction) <-- this one seems to perform the best, though the differences are small

# Requirements

- `python3`
- `pip3 install -r requirements.txt`

# Usage

Below are three example scenarios for using this project.

## Train a new model and evaluate

`python run_ner.py --data_dir=data/bio --bert_model=bert-base-cased --output_dir=out_base --max_seq_length=64 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`

This example uses the `bert-base-cased` model that is hosted by Huggingface, which will be downloaded automatically if neccessary. You can also use a local model by supplying a path to the `--bert_model` argument.

## Run leave-one-out evaluation

`python run_ner_patent_clean.py --data_dir=data/bio --bert_model=bert-base-cased --max_seq_length=64 --do_leave_one_out --num_train_epochs 5 --do_eval --warmup_proportion=0.1 --output_dir=out_leave_one_out`

## Use a finetuned model on new data

`python run_ner_patent_clean.py --data_dir=data/new_data --bert_model=./path/to/model --task_name=ner --output_dir=out_results --max_seq_length=64 --do_eval --train_ratio=0`

Note the `--train_ratio=0`, meaning none of the data files will be kept separate for training, and evaluation is run on all files.