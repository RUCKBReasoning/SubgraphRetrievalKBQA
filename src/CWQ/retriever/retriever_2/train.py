from transformers import AutoConfig, TrainingArguments
from trainer import CLTrainer as Trainer
import torch
import datasets
import argparse
from transformers import AutoTokenizer
from model import BertForCL, RobertaForCL
import os

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path',
                    default='bert-base-uncased', type=str)
parser.add_argument('--train_dataset',
                    default='./train_tokenized.jsonl', type=str)
parser.add_argument(
    '--eval_dataset', default='./dev_tokenized.jsonl', type=str)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--per_device_train_batch_size', default=16, type=int)
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--num_train_epochs', default=10, type=int)
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--fp16', default=True, type=bool)
parser.add_argument('--output_dir', default='./output_dir_v2', type=str)
parser.add_argument('--logging_dir', default='./log_dir_v2', type=str)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--temperature', default=0.07, type=float)
args = parser.parse_args()

# %%
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

train_data = datasets.load_dataset("json", data_files={
    "train": args.train_dataset}, cache_dir="./cache")["train"]


eval_data = datasets.load_dataset("json", data_files={
    "eval": args.eval_dataset}, cache_dir="./cache")["eval"]


# %%
encoder_config = AutoConfig.from_pretrained(args.pretrained_model_name_or_path)
if encoder_config.architectures[0] == 'RobertaModel':
    model = RobertaForCL(encoder_config, temp=args.temperature)
elif encoder_config.architectures[0] == 'BertModel':
    model = BertForCL(encoder_config, temp=args.temperature)
else:
    raise NotImplementedError
print('Load from ckpt: ', args.pretrained_model_name_or_path)
with open(os.path.join(args.pretrained_model_name_or_path, 'pytorch_model.bin'), 'rb') as f:
    model.encoder.load_state_dict(torch.load(f), strict=True)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)


train_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    warmup_steps=500,
    weight_decay=0.01,               # strength of weight decay
    logging_dir=args.logging_dir,            # directory for storing logs
    logging_steps=100,
    local_rank=args.local_rank,
    evaluation_strategy='steps',
    eval_steps=args.eval_steps,
    eval_accumulation_steps=1,
    fp16=args.fp16,
    sharded_ddp=True,
    learning_rate=args.learning_rate,
)


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
)
trainer.train()
