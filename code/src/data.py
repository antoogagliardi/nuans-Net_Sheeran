import re
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import BertTokenizer
from tqdm import tqdm
from typing import List




class LyricDataset(Dataset):
    def __init__(self, in_data:dict, gold_data:dict, tokenizer:BertTokenizer):
        super(LyricDataset, self).__init__()
        self.encoding = {}
        self.out_labels = {}
        with tqdm(range(len(in_data)), desc="DataLoader Started") as pbar:
            for i, line1, line2s in zip(pbar, in_data.keys(), in_data.values()):
                self.build_model_input_data(line1, line2s, gold_dict=gold_data, tokenizer=tokenizer)
        
        print("encoding length: ", len(self.encoding["input_ids"]))
        print("ground truth encoding shape: ", len(self.out_labels["mask_ground_truth"]))
        print("next sentence prediction labels length: ", len(self.out_labels["next_sentence_label"]))
    
    
    def mask_sentences(self, tokenizer:BertTokenizer, sentA:str=None, sentB:str=None, n_elem:int=1):
        forbidden_idx = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
        mask_sentA = re.split(" ", sentA); mask_sentB = re.split(" ", sentB)                                    # Create what will become "masked sentences"
        if sentA == None or sentB == None: print("Error")
        else:
            # Create lists of candidates index to be masked
            idx_choiceA = [idx for idx in range(len(mask_sentA)) if mask_sentA[idx] not in forbidden_idx]
            idx_choiceB = [idx for idx in range(len(mask_sentB)) if mask_sentB[idx] not in forbidden_idx]   
            # Choose random indices from the lists of possible indices
            choiceA = random.sample(idx_choiceA, n_elem if len(idx_choiceA)>1 else 1)
            choiceB = random.sample(idx_choiceB, n_elem if len(idx_choiceB)>1 else 1)
            
            for a, b in zip(choiceA, choiceB):
                mask_sentA[a] = tokenizer.mask_token
                mask_sentB[b] = tokenizer.mask_token
        
            return " ".join(mask_sentA), " ".join(mask_sentB)
        
    def build_model_input_data(self, sentenceA:str, sentenceBs:List[str], gold_dict:dict, tokenizer:BertTokenizer):
        for sentence_B in sentenceBs:
            mask_sentA, mask_sentB = self.mask_sentences(tokenizer=tokenizer,
                                                         sentA=sentenceA, sentB=sentence_B, n_elem=1)
            encoding_i = tokenizer(sentenceA, sentence_B, add_special_tokens=True,
                                   padding="max_length", truncation=True)
            mask_encoding_i = tokenizer(mask_sentA, mask_sentB, add_special_tokens=True,
                                        padding="max_length", truncation=True)
            
            mask_encoding_i["attention_mask"] = list(mask_encoding_i["attention_mask"])
            mask_encoding_i["input_ids"] = list(mask_encoding_i["input_ids"])
            mask_encoding_i["token_type_ids"] = list(mask_encoding_i["token_type_ids"])
            # 0: Sentence_B is one of the possibile random sentences that can follow Sentence_A
            # 1: Sentence_B follow naturally Sentence_A
            out_labels_i = {"mask_ground_truth": list(encoding_i["input_ids"]),
                            "next_sentence_label": [1] if gold_dict[f"{sentenceA}"] == sentence_B else [0]}
            
            if len(self.encoding) == 0:
                self.encoding["attention_mask"] = [mask_encoding_i["attention_mask"]]
                self.encoding["input_ids"] = [mask_encoding_i["input_ids"]]
                self.encoding["token_type_ids"] = [mask_encoding_i["token_type_ids"]]

                self.out_labels["mask_ground_truth"] = [out_labels_i["mask_ground_truth"]]
                self.out_labels["next_sentence_label"] = [out_labels_i["next_sentence_label"]]
            else:
                self.encoding["attention_mask"].append(mask_encoding_i["attention_mask"])
                self.encoding["input_ids"].append(mask_encoding_i["input_ids"])
                self.encoding["token_type_ids"].append(mask_encoding_i["token_type_ids"])

                self.out_labels["mask_ground_truth"].append(out_labels_i["mask_ground_truth"])
                self.out_labels["next_sentence_label"].append(out_labels_i["next_sentence_label"])
    
    def __len__(self):
        if len(self.encoding["input_ids"]) == len(self.out_labels["next_sentence_label"]) and \
            len(self.encoding["input_ids"]) == len(self.out_labels["mask_ground_truth"]): return len(self.encoding["input_ids"])
        else: print("Errore nella lettura della lunghezza del Dataset")

    def __getitem__(self, idx):
        tokens = self.encoding["input_ids"][idx]
        tokens_type = self.encoding["token_type_ids"][idx]
        attention_mask = self.encoding["attention_mask"][idx]

        mask_ground_truth = self.out_labels["mask_ground_truth"][idx]
        nsp_labels = self.out_labels["next_sentence_label"][idx]
        
        return (tokens, tokens_type, attention_mask, mask_ground_truth, nsp_labels)




class DataManager(pl.LightningDataModule):
    def __init__(self, train_data:Dataset=None, valid_data:Dataset=None, test_data:Dataset=None):
        super(DataManager, self).__init__()
        if train_data != None: self.train_data = train_data
        if valid_data != None: self.valid_data = valid_data
        if test_data != None: self.test_data = test_data
    
    def add_train_data(self, train_data:Dataset):
        self.train_data = train_data
    def add_valid_data(self, valid_data:Dataset):
        self.valid_data = valid_data
    def add_test_data(self, test_data:Dataset):
        self.test_data = test_data

    def train_dataloader(self, batch_size:int, drop_last:bool=False, suffle_data:bool=True):
        train_dataloader = DataLoader(self.train_data, batch_size=batch_size,
                                      shuffle=suffle_data, collate_fn=self.collate_data,
                                      drop_last=drop_last)
        return train_dataloader
    
    def val_dataloader(self, batch_size:int, drop_last:bool=False):
        valid_dataloader = DataLoader(self.valid_data, batch_size=batch_size,
                                      shuffle=False, collate_fn=self.collate_data,
                                      drop_last=drop_last)
        return valid_dataloader
    
    def test_dataloader(self, batch_size:int, drop_last:bool=False):
        test_dataloader = DataLoader(self.test_data, batch_size=batch_size,
                                     shuffle=False, collate_fn=self.collate_data,
                                     drop_last=drop_last)
        return test_dataloader
    
    def collate_data(self, samples):
        tokens = torch.LongTensor([sample[0] for sample in samples])
        tokens_type = torch.LongTensor([sample[1] for sample in samples])
        attention_mask = torch.LongTensor([sample[2] for sample in samples])

        mask_truth_labels = torch.LongTensor([sample[3] for sample in samples]) 
        nsp_labels = torch.LongTensor([sample[4] for sample in samples])   
        
        return {"tokens": tokens,
                "tokens_type": tokens_type,
                "attention_mask": attention_mask,
                "mask_ground_truth": mask_truth_labels,
                "nsp_labels": nsp_labels}
