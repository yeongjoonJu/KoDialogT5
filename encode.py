import argparse
import json, os, re
import sys
from tqdm import tqdm
from glob import glob
import numpy as np
from os.path import join as pjoin
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

def create_training_data(examples, tokenizer, args, filename, special_token_ids):
    usr_bos_id, usr_eos_id, sys_bos_id, sys_eos_id = special_token_ids
    print("\nDataset:", filename)
    print()
    mean_res_len = 0
    total_res_cnt = 0
    total_samples = len(examples)
    results = []
    for e, example in enumerate(examples):
        dialog = []

        for i, turn in enumerate(example):
            usr = turn['usr'].lower()
            usr = re.sub(r" ##", "", usr)
            usr = tokenizer.encode(usr)[:-1]
            usr = [usr_bos_id] + usr + [usr_eos_id]
            sys = turn['sys'].lower()
            sys = re.sub(r" ##", "", sys)
            sys = tokenizer.encode(sys)[:-1]
            sys = [sys_bos_id] + sys + [sys_eos_id]

            dialog.append({'usr':usr, 'sys':sys})

            mean_res_len += len(sys)
            total_res_cnt += 1

        cur = e+1
        print("\r [%s: %d/%d (%.2f%%)] / Mean Length [Res] %.2f" % (filename, cur, total_samples, \
                (cur/total_samples)*100, mean_res_len/total_res_cnt), end="")

        results.append(dialog)
    print()
        
    return results


def main(args):
    tokenizer = T5Tokenizer.from_pretrained(args.backbone, do_lower_case=False, cache_dir=None)

    special_tokens = ['<usr>', '</usr>', '<sys>','</sys>']
    tokenizer.add_tokens(special_tokens)

    usr_bos_id = tokenizer.convert_tokens_to_ids(['<usr>'])[0]
    usr_eos_id = tokenizer.convert_tokens_to_ids(['</usr>'])[0]
    sys_bos_id = tokenizer.convert_tokens_to_ids(['<sys>'])[0]
    sys_eos_id = tokenizer.convert_tokens_to_ids(['</sys>'])[0]
    special_token_ids = (usr_bos_id, usr_eos_id, sys_bos_id, sys_eos_id)

    datasets = []
    files = glob(pjoin(args.data_path, "*.json"))
    files = sorted(files, reverse=True)
    print("<Using datasets>")
    print(files)

    for filename in files:
        data = json.load(open(filename))
        data = create_training_data(data, tokenizer, args, filename, special_token_ids=special_token_ids)
        datasets.extend(data)

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(args.data_path, "encoded")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, args.data_name+'_id.json'), 'w', encoding='utf-8') as fout:
        json.dump(datasets, fout)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="digit82/kolang-t5-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--data_name", type=str, default="kor_merge")

    args = parser.parse_args()
    main(args)