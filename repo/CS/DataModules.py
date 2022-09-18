import re

import pandas as pd
from bs4 import BeautifulSoup
from torch.utils.data import Dataset

from Constants import *

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    text = re.sub(r'[\\]', ' ', text)
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, regex_transformations={}):
        # Read JSON file and assign to data variable (list of strings)
        # df = pd.read_json(dataset_file_path, lines=True)
        # df = df.drop(['article_link'], axis=1)
        self.data = pd.read_csv(dataset_file_path)
        # Apply function on review column
        self.data['Description'] = self.data['Description'].apply(denoise_text)
        #self.data['label'] = self.data['sentiment']

        # self.data = df.values
        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.num_class = len(self.data['Class Index'].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = int(self.data['Class Index'][index]) - 1
        text = self.data['Description'][index]
        for regex, value_to_replace_with in self.regex_transformations.items():
            text = re.sub(regex, value_to_replace_with, text)

        # Convert input string into tokens with the special BERT Tokenizer which can handle out-of-vocabulary words using subgrams
        # e.g. text = Here is the sentence I want embeddings for.
        #      tokens = [here, is, the, sentence, i, want, em, ##bed, ##ding, ##s, for, .]
        tokens = self.tokenizer.tokenize(text)[:MAX_SEQ_LENGTH-3]

        # Add [CLS] at the beginning and [SEP] at the end of the tokens list for classification problems
        tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
        # Convert tokens to respective IDs from the vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Segment ID for a single sequence in case of classification is 0.
        segment_ids = [0] * len(input_ids)

        # Input mask where each valid token has mask = 1 and padding has mask = 0
        input_mask = [1] * len(input_ids)

        # padding_length is calculated to reach max_seq_length
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        input_mask = input_mask + [0] * padding_length
        segment_ids = segment_ids + [0] * padding_length

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH

        return torch.tensor(input_ids, dtype=torch.long, device=DEVICE), \
               torch.tensor(segment_ids, dtype=torch.long, device=DEVICE), \
               torch.tensor(input_mask, device=DEVICE), \
               torch.tensor(label, dtype=torch.long, device=DEVICE)