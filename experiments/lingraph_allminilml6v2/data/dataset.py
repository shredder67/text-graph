from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

from .graph_utils import linearize_graph
    
class TextGraphDataset(Dataset):
    def __init__(
        self,  
        tokenizer, 
        max_length: int, 
        train_path: str,
        test_path: str,
        split: str='train',
        include_graph: bool=False
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
        if split in ['train', 'val', 'test']:
            df = pd.read_csv(train_path, sep='\t')
            df["label"] = df["correct"].astype(np.float32)
            self.df = self._split_train_dev_test(df, split)
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
            a_entities = self.graphs[idx]
            
         
        tokenizer_out = self.tokenizer.encode_plus(
                text=q_entities + ' ' + question + '[SEP]' + a_entities,
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
            self.graphs = self.df["graph"].apply(linearize_graph).to_list()
            
    def _split_train_dev_test(self, df, split='train'):
        all_questions = list(df["question"].unique())
        num_questions = len(all_questions)
        random.shuffle(all_questions)

        train_dev_ratio = 0.8
        train_ratio = 0.9
        num_train_dev_questions = int(num_questions * train_dev_ratio)
        train_dev_questions = all_questions[:num_train_dev_questions]
        test_questions = set(all_questions[num_train_dev_questions:])
        num_train_questions = int(len(train_dev_questions) * train_ratio)
        train_questions = set(train_dev_questions[:num_train_questions])
        dev_questions = set(train_dev_questions[num_train_questions:])

        train_df = df[df["question"].isin(train_questions)]
        dev_df = df[df["question"].isin(dev_questions)]
        test_df = df[df["question"].isin(test_questions)]

        if split == 'train':
            return train_df
        elif split =='val':
            return dev_df
        else:
            return test_df