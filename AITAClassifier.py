import torch
from torch import nn, optim

class AitaClassifier(nn.Module):

    def __init__(self, n_classes):
        super(AitaClassifier, self).__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        output = self.drop(pooled_output)
        return self.out(output)
