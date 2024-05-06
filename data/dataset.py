from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

from .graph_utils import *
    
class TextGraphDataset(Dataset):
    def __init__(
        self,  
        tokenizer, 
        max_length: int, 
        train_path: str,
        test_path: str,
        split: str,
        df_split: pd.DataFrame,
        include_graph: bool=False,
        is_T5: bool=False,
    ):
        """
        include_graph
            if False: __getitem__() returns q_entities+' '+question+'[SEP]'+a_entities
            if True: __getitem__() returns q_entities+' '+question+'[SEP]'+linearized_graph
                linearized_graph contains a_entities with [SEP] surroundings
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.include_graph = include_graph
        self.is_T5 = is_T5
        if split in ['train', 'val', 'test']:
            self.df = df_split
        elif split == 'full': # use this to use all data for training (before submit)
            self.df = pd.read_csv(train_path, sep='\t')
            self.df["label"] = self.df["correct"].astype(np.float32)
        elif split == 'eval': # this corresponds to submit
            self.df = pd.read_csv(test_path, sep='\t')
        else:
            raise ValueError("Unrecognized split!")
        
        self.questions = []
        self.q_entities = []
        self.a_entities = []
        self.graphs = []
        self.labels = []

        self._get_data()

    def __getitem__(self, idx):
        q_entities = self.q_entities[idx] + ':'
        question = self.questions[idx]
        a_entities = self.a_entities[idx]
        
        if self.include_graph:
            if self.is_T5:
                a_entities = self.graphs[idx] + ' ' + a_entities
                tokenizer_in_text = 'predict [SEP] ' + question + '[SEP]' + a_entities    
            else:
                a_entities = self.graphs[idx]
                tokenizer_in_text = q_entities + ' ' + question + '[SEP]' + a_entities
        else: 
                tokenizer_in_text = q_entities + ' ' + question + '[SEP]' + a_entities
            
        tokenizer_out = self.tokenizer.encode_plus(
                text=tokenizer_in_text,
                max_length=self.max_length,
                padding="max_length",
                truncation="only_first",
                return_tensors="pt"
            )
        
        res = {
            "input_ids": tokenizer_out["input_ids"].flatten(),
            "attention_mask": tokenizer_out["attention_mask"].flatten(),
        }
        
        if self.split != "eval":
            res["labels"] = self.labels[idx]
        
        if "token_type_ids" in tokenizer_out:
            res["token_type_ids"] = tokenizer_out["token_type_ids"].flatten()
        
        return res

    def __len__(self):
        return len(self.df)
    
    def _get_data(self):
        self.questions = self.df["question"].to_list()
        self.q_entities = self.df["questionEntity"].to_list()
        self.a_entities = self.df["answerEntity"].to_list()
        if self.split != "eval":
            self.labels = self.df["label"].to_list()
        if self.include_graph:
            self.df["graph"] = self.df["graph"].apply(eval)
            if self.is_T5:
                self.graphs = self.df["graph"].apply(linearize_graph_T5).to_list()
            else:
                self.graphs = self.df["graph"].apply(linearize_graph).to_list()
            