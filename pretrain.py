from __future__ import absolute_import, division, print_function

import argparse
import random, os
import numpy as np

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

from models import T5forDialog, Runner
from dataset import PretrainingDataBuilder

from torch.utils.data import DataLoader

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def collate_wrapper(batch):
    batch = list(zip(*batch))
    sample = {}
    
    # encoder
    enc_max_len = max([x.size(0) for x in batch[0]])
    enc_inputs = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
    enc_attn_mask = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.float)
    for i, x in enumerate(batch[0]):
        enc_inputs[i,:x.size(0)] = x
        enc_attn_mask[i,:x.size(0)] = 1.0
    
    # response
    res_max_len = max([x.size(0) for x in batch[1]])
    resp_label = torch.zeros((len(batch[1]), res_max_len), dtype=torch.long)
    for i, x in enumerate(batch[1]):
        resp_label[i,:x.size(0)] = x

    sample['enc_inputs'] = enc_inputs
    sample['enc_attn_mask'] = enc_attn_mask
    sample['resp_label'] = resp_label
    
    return sample

def main(args):
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    tokenizer = T5Tokenizer.from_pretrained(args.backbone)
    config = T5Config.from_pretrained(args.backbone)

    resized_vocab_size = config.vocab_size
    special_tokens = ["<usr>", "</usr>", "<sys>", "</sys>"]
    if args.add_special_action_tokens:
        for line in open(args.add_special_action_tokens):
            line = line.strip()
            if not line in special_tokens:
                special_tokens.append(line)
    
    tokenizer.add_tokens(special_tokens)
    resized_vocab_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    print("Tokenizer len:", len(tokenizer))
    runner = Runner(config, args=args, resized_vocab_size=len(tokenizer))
    runner.set_tokenizer(tokenizer)

    # Load dataset
    builder = PretrainingDataBuilder(config, enc_max_seq=args.enc_max_seq, dec_max_seq=args.dec_max_seq, max_turn=args.max_turn)
    
    actual_train_batch_size = args.train_batch_size*args.n_gpu*args.gradient_accumulation_steps

    if args.valid_file is not None:
        train_dataset = builder.get_dataset(tokenizer, args.train_file, cached_path=args.cached_train_data, \
                                            batch_size=actual_train_batch_size, valid=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.n_workers, shuffle=False, collate_fn=collate_wrapper) 
        valid_dataset = builder.get_dataset(tokenizer, args.valid_file, cached_path=args.cached_valid_data, valid=True)
        val_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.n_workers//2, shuffle=False, collate_fn=collate_wrapper)
    else:
        train_dataset, valid_dataset = builder.get_dataset(tokenizer, args.train_file, cached_path=args.cached_train_data, \
                                                        batch_size=actual_train_batch_size, valid=False, split=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.n_workers, shuffle=False, collate_fn=collate_wrapper)
        val_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.n_workers//2, shuffle=False, collate_fn=collate_wrapper)

    if args.max_steps > 0:
        args.num_training_steps = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        args.num_training_steps = int(len(train_dataloader) * args.num_train_epochs / args.n_gpu)

    if args.warmup_steps < 0:
        args.warmup_steps = int(len(train_dataloader) / args.n_gpu * args.num_train_epochs * 0.2)

    val_iters = len(train_dataloader)//args.n_gpu//3
    print("Val iters:", val_iters, "Warmup steps:", args.warmup_steps)

    # declare trainer
    os.environ['WANDB_CONSOLE'] = 'off'
    os.environ['WAND_DISABLE'] = 'True'
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_PPL', mode='min', dirpath=args.output_dir, \
                                        save_top_k=3, filename="{epoch}-{val_PPL:.3f}")
    wandb_logger = WandbLogger(project="KorDialogT5" ,name=args.log_name)

    trainer = pl.Trainer(gpus=args.n_gpu, max_epochs=args.num_train_epochs, accumulate_grad_batches=args.gradient_accumulation_steps, \
                        callbacks=[checkpoint_callback], check_val_every_n_epoch=1, val_check_interval=val_iters, \
                        strategy=args.distributed_strategy if args.distributed_strategy!='none' else None,  \
                        precision=16 if args.fp16 else 32, gradient_clip_val=args.max_grad_norm, \
                        enable_progress_bar=True, default_root_dir=args.output_dir, logger=wandb_logger)

    trainer.fit(runner, train_dataloader, val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", type=str, default=None, required=True,
                        help="The i nput labeled training data files (json file).")
    parser.add_argument("--valid_file", type=str, default=None, required=False,
                        help="The input validation data file (json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--distributed_strategy", type=str, default='ddp', required=False, help='such as deepspeed_stage_2 and deepspeed_stage_3')

    ## Other parameters
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--log_name", default="", type=str)
    parser.add_argument("--backbone", default="KETI-AIR/ke-t5-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--resume", default="", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--cached_train_data", type=str, default="")
    parser.add_argument("--cached_valid_data", type=str, default="")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--add_special_action_tokens", default='', type=str)
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--valid_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for validation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--seed', type=int, default=11,
                        help="random seed for initialization")
    parser.add_argument("--optimizer", type=str, default="adamw", help="adamw or adafactor")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4, help="")
    parser.add_argument("--enc_max_seq", default=512, type=int, help="")
    parser.add_argument("--dec_max_seq", default=80, type=int, help="")
    parser.add_argument("--max_turn", default=15, type=int, help="")

    args = parser.parse_args()

    main(args)

    