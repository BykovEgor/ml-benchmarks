import glob
import logging
import os
import typing as t

import torch
import torch.utils.data as tud
import transformers

T = t.TypeVar('T')

logging.basicConfig(level=logging.INFO)


class ArticlesDataset(tud.Dataset):
    """News Articles Dataset"""
    
    def __init__(self, data_dir: str, file_name_glob: str, transform: bool = False):
        
        self.files = glob.glob(os.path.join(data_dir, file_name_glob))
        self.filenum = len(self.files)
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transform
        
        with open(self.files[0], "r") as f:
            sample = f.readlines()
        
        self.linenum = len(sample)
    
    def __len__(self):
        
        return self.linenum * self.filenum
    
    def __getitem__(self, idx):
        
        filenum = (idx + 1) // self.linenum - 1
        linenum = ((idx + 1) % self.linenum) - 1
        
        with open(self.files[filenum], "r") as f:
            sample = f.read().split("\n")
            
            if self.transform:
                sample = self.tokenize(sample)
            
            output = dict()
            
            for key in sample:
                output[key] = torch.unsqueeze(sample[key][linenum], 0)
            
            return output
    
    def tokenize(self, text):
        
        return self.tokenizer(text, return_tensors='pt', padding=True)


def bertBatchCollate(batch: t.List[T]) -> t.Any:
    output = dict()
    for key in batch[0]:
        output[key] = torch.stack([el[key] for el in batch])
        x, y, z = output[key].size()
        output[key] = output[key].view(x * y, z)
    
    return output
