import os
from torch.utils.data import Dataset
import yaml
from transformers import AutoTokenizer
import torch

class corpus_dataset(Dataset):
    def __init__(self, data_dir):
        if data_dir == 'data_dir':
            self.questions, self.labels, answer, label_dict = read_questions_from_dir(data_dir)
        else:
             self.questions, self.labels, label_dict = read_questions_from_dir(data_dir)
        self.token_ids = self.questions['input_ids']
        self.attn_masks =  self.questions['attention_mask']
        self.token_type_ids = self.questions['token_type_ids']

    def __len__(self):
        return len(self.token_ids)
    def __getitem__(self,i):
        token_ids = torch.tensor(self.token_ids[i])
        attn_masks = torch.tensor(self.attn_masks[i]) # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids =  torch.tensor(self.token_type_ids[i]) # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        labels = torch.tensor(self.labels[i])
        return token_ids,attn_masks,token_type_ids,labels

def read_questions_from_dir(data_dir):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    questions = []
    answer = []
    labels = []
    label_dict = {}
    labelcode = 0
    for file in os.listdir(data_dir):
        yamlPath = os.path.join(data_dir, file)

        f = open(yamlPath, 'r', encoding='utf-8')
        cfg = f.read()
        corpus = yaml.load(cfg, Loader=yaml.FullLoader)
        if data_dir == 'data_dir':
            answer.append(corpus['answers'])
        questions.append(corpus['questions'])
        l = [labelcode] * len(corpus['questions'])
        label_dict[labelcode] = corpus['categories']
        labels.append(l)
        labelcode+=1
   
    q = ['CLS' + token + 'SEP' for s in questions for token in s]
    label = [l for z in labels for l in z]

    encoded_questions = tokenizer(q, padding=True)
    if data_dir == 'data_dir':
        return encoded_questions, label, answer, label_dict
    else:
        return encoded_questions, label, label_dict
