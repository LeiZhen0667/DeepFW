import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def load_pretrained_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            model[word] = vector
    return model


def preprocess_text_for_html(text):
    # cleaned_text = re.sub(r'[^\w\s]', '', text)  # 移除所有的标点符号
    cleaned_text = text.replace('\n', ' ')  # 移除换行符
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # 移除连续两个空格以上的空格组合
    return cleaned_text


def get_web_files(root_folder):
    web_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(('.html', '.htm', '.php', '.asp', '.js', '.css')):
                web_files.append(file_path)
    return web_files


def filter_empty_files(files):
    non_empty_files = []
    for file_path in files:
        if os.path.getsize(file_path) > 0:
            non_empty_files.append(file_path)
    return non_empty_files


class TextWithLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model, file_names):
        self.texts = texts
        self.labels = labels
        self.file_names = file_names
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_dim = 300
        self.word_index = self.create_word_index()
        self.embedding_matrix = self.create_embedding_matrix()

    def create_word_index(self):
        word_index = {}
        for i, token in enumerate(self.tokenizer.get_vocab().keys()):
            word_index[token] = i
        return word_index

    def create_embedding_matrix(self):
        embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dim))
        for word, i in self.word_index.items():
            if word in self.model:
                embedding_matrix[i] = self.model[word]
        return torch.tensor(embedding_matrix, dtype=torch.float32)

    def texts_to_sequences(self, texts):
        sequences = [torch.tensor(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        sequences = [seq for seq in sequences if len(seq) > 0]
        if not sequences:
            sequences = [torch.tensor([0])]
        return sequences

    def pad_sequences(self, sequences, max_len=512):
        padded_sequences = []
        for seq in sequences:
            if len(seq) == 0:
                continue
            seq = torch.tensor(seq)
            num_chunks = (len(seq) + max_len - 1) // max_len
            for i in range(num_chunks):
                chunk = seq[i * max_len:(i + 1) * max_len]
                if len(chunk) < max_len:
                    chunk = torch.cat([chunk, torch.zeros(max_len - len(chunk), dtype=torch.long)])
                padded_sequences.append(chunk)
        if len(padded_sequences) == 0:
            return torch.zeros((1, max_len), dtype=torch.long)
        return torch.stack(padded_sequences)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        file_name = self.file_names[idx]
        sequences = self.texts_to_sequences([text])
        if not sequences:
            sequences = [torch.tensor([0])]
        padded_sequences = self.pad_sequences(sequences)
        return padded_sequences, label, file_name

def collate_fn(batch):
    sequences, labels, file_names = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels)
    return sequences_padded, labels_tensor, file_names
