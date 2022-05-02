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

def interaction(args):
    config = T5Config.from_pretrained(args.backbone)
    special_tokens = ['<usr>', '</usr>', '<sys>','</sys>']
    tokenizer = T5Tokenizer.from_pretrained(args.backbone, do_lower_case=False, cache_dir=None)
    tokenizer.add_tokens(special_tokens)
    model = Runner.load_from_checkpoint(args.resume, strict=False, config=config, args=args, resized_vocab_size=len(tokenizer)).model    
    model = model.to(args.gpu_id)

    SYS_BOS_ID, SYS_EOS_ID = tokenizer.convert_tokens_to_ids(['<sys>','</sys>'])
    UNK_ID = tokenizer.convert_tokens_to_ids(["<unk>"])[0]
    EOS_ID = tokenizer.eos_token_id

    print("\nSystem: 안녕하세요!")
    context = []
    while True:
        uttr = input("\nUser: ")
        if uttr=="exit":
            break
        uttr = f"<usr> {uttr.strip()} </usr>"
        uttr = tokenizer.encode(uttr)
        context.extend(uttr)

        enc_inputs = torch.LongTensor(context).unsqueeze(0).to(args.gpu_id)
        attn_mask = [1.]*len(context)
        attn_mask = torch.FloatTensor(attn_mask).unsqueeze(0).to(args.gpu_id)
        response = model.generate(input_ids=enc_inputs,
                                attention_mask=attn_mask,
                                eos_token_id=EOS_ID,
                                max_length=200,
                                early_stopping=True,
                                temperature=1.0)
        response = response.cpu().numpy().tolist()[0]
        response = response[1:-1]
        context = context[:-1]
        context.extend([SYS_BOS_ID] + response + [SYS_EOS_ID])
        response = tokenizer.decode(response, clean_up_tokenization_spaces=True)
        print("\nSystem:", response)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="digit82/kolang-t5-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--resume", default="checkpoints/KorDial_raw/epoch=9-val_PPL=11.063.ckpt", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--gpu_id", default=0, type=int)
    args = parser.parse_args()
    
    interaction(args)

    
