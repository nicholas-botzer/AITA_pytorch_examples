from torch.utils.data import Dataset, DataLoader
import torch


class AitaDataset(Dataset):

    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = self.comments[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(comment, truncation=True, max_length=self.max_len, add_special_tokens=True,
                                              pad_to_max_length=True, return_tensors='pt')

        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }