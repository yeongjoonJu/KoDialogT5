from matplotlib.pyplot import magnitude_spectrum
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import os, pickle, json
import random, math
from glob import glob
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, Dataset
from transformers import T5Config, T5Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    def __init__(self, enc_inputs, resp_label):
        self.enc_inputs = enc_inputs
        self.resp_label = resp_label
    
    def get_data(self):
        return {"enc_inputs":self.enc_inputs, "resp_label":self.resp_label,}
    
    def __getitem__(self, index):
        return self.enc_inputs[index], self.resp_label[index]
    
    def __len__(self):
        return len(self.enc_inputs)


class PretrainingDataBuilder():
    def __init__(self, config, enc_max_seq=512, dec_max_seq=512, max_turn=15):
        self.max_turn = max_turn
        self.enc_max_seq = enc_max_seq
        self.dec_max_seq = dec_max_seq

    def get_dataset(self, tokenizer, file_path, cached_path='data/cached_multiwoz.json', batch_size=1, valid=False, split=False):
        if os.path.exists(cached_path):
            data = torch.load(cached_path)
            # with open(cached_path, "r") as fin:
            #     data = json.load(fin)
            dataset = PretrainingDataset(data["enc_inputs"], data["resp_label"])
            if split:
                valid_cached_path = '.'.join(cached_path.split('.')[:-1])+'_valid.json'
                valid_data = torch.load(valid_cached_path)
                valid_dataset = PretrainingDataset(valid_data["enc_inputs"], valid_data["resp_label"])
                return dataset, valid_dataset
        else:
            self.vocab_size = len(tokenizer)

            # Define special token id
            self.eos_id = tokenizer.eos_token_id
            self.pad_id = tokenizer.pad_token_id
            self.usr_start_id = tokenizer.convert_tokens_to_ids(['<usr>'])[0]
            self.usr_end_id = tokenizer.convert_tokens_to_ids(['</usr>'])[0]
            self.sys_start_id = tokenizer.convert_tokens_to_ids(['<sys>'])[0]
            self.sys_end_id = tokenizer.convert_tokens_to_ids(['</sys>'])[0]
            # self.extra_ids = [ tokenizer._convert_token_to_id(f'<extra_id_{n}>') for n in range(tokenizer._extra_ids) ]

            logger.info(f"Loading dialogues")
            with open(file_path, 'r') as f:
                dialogs = json.load(f)

            n_dialogs = len(dialogs)
            logger.info("{} dialogues were loaded".format(n_dialogs))

            if split:
                random.shuffle(dialogs)
                valid_data = dialogs[:2000]
                dialogs = dialogs[2000:]

            data = self.prepare_data(dialogs, batch_size=batch_size, valid=valid)
            dataset = self.caching(data)
            if split:
                valid_data = self.prepare_data(valid_data, batch_size=batch_size, valid=True)
                valid_dataset = self.caching(valid_data)

            logger.info('Saving cached data...')
            cached_data = dataset.get_data()
            torch.save(cached_data, cached_path)
            if split:
                valid_cached_path = '.'.join(cached_path.split('.')[:-1])+'_valid.json'
                cached_valid_data = valid_dataset.get_data()
                torch.save(cached_valid_data, valid_cached_path)

                return dataset, valid_dataset

        return dataset

    def caching(self, data):
        enc_inputs = []
        resp_label = []

        for obj in tqdm(data):
            enc_inputs.append(torch.LongTensor(obj['enc_inputs']))
            resp_label.append(torch.LongTensor(obj['resp_label']))
        
        dataset = PretrainingDataset(enc_inputs, resp_label)

        return dataset


    def build_training_dialog(self, dialog, valid=False):
        enc_inputs = []
        resp_label = []
        
        n_turns = len(dialog)
        start_turn = 0
        context_len = 2
        for i in range(n_turns-1, -1, -1):
            cur_turn_len = len(dialog[i]["usr"])+len(dialog[i]["sys"])
            if context_len + cur_turn_len > self.enc_max_seq:
                start_turn = i+1
                break
            context_len += cur_turn_len

        for i in range(start_turn, n_turns-1):
            sess = dialog[i]
            # User utterance
            enc_inputs.extend(sess['usr'])
            enc_inputs.extend(sess['sys'])
        
        # response processing
        enc_inputs.extend(dialog[n_turns-1]["usr"])
        enc_inputs.append(self.eos_id)
        system = dialog[n_turns-1]['sys']
        resp_label.extend(system[1:-1])
        resp_label.append(self.eos_id)
        
        if len(resp_label) > self.dec_max_seq:
            resp_label = resp_label[:self.dec_max_seq]
        
        output =  {'enc_inputs': enc_inputs, 'resp_label': resp_label}
        
        return output


    def prepare_data(self, dialogs, batch_size=1, valid=False):
        data = []
        for index in tqdm(range(len(dialogs))):
            dialog = dialogs[index]
            dialog = dialog[-self.max_turn:]

            processed = self.build_training_dialog(dialog, valid)
            processed.update({"n_turns": len(dialog)})
            data.append(processed)

        random.shuffle(data)
        if batch_size > 1:
            data = sorted(data, key=lambda x: x["n_turns"], reverse=False)
            total_batch_samples = []
            batch_samples = []
            for i in range(len(data)):
                batch_samples.append(data[i])
                if len(batch_samples) >= batch_size:
                    total_batch_samples.append(batch_samples.copy())
                    batch_samples = []
            if len(batch_samples)!=0:
                total_batch_samples.append(batch_samples.copy())
            
            random.shuffle(total_batch_samples)
            new_data = []
            for batch_s in total_batch_samples:
                new_data.extend(batch_s)
            data = new_data

        return data