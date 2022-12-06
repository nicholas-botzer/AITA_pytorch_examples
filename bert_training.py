import torch
import transformers
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from held_out_evaluation import get_binary_evaluation_metrics, write_binary_evaluation_output
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from AITADataset import AitaDataset
from AITAClassifier import AitaClassifier
from tqdm import tqdm
from sklearn.model_selection import KFold


def get_label(label):

    if "YTA" in label  or "ESH" in label:
        lbl_val = 1
    else:
        lbl_val = 0

    return lbl_val


def create_data_loader(df, tokenizer, max_len, batch_size):

    ds = AitaDataset(
        comments=df.body.to_numpy(),
        labels=df.labels.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        # print(f'Predictions: {preds}')
        # print(f'Labels: {labels}')

        correct_predictions += torch.sum(preds == labels)
        # print(f'correct predictions: {correct_predictions}')
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):

    model = model.eval()

    comment_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["comment_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            comment_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return comment_texts, predictions, prediction_probs, real_values


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
      for d in data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          labels = d["labels"].to(device)

          outputs = model(input_ids=input_ids, attention_mask=attention_mask)

          _, preds = torch.max(outputs, dim=1)

          loss = loss_fn(outputs, labels)

          correct_predictions += torch.sum(preds == labels)

          losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def show_confusion_matrix(confusion_matrix):
    dims = (12,12)
    plt.subplots(figsize=dims)
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

    plt.ylabel('True Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.savefig("confusion_matrix.png")

RANDOM_SEED = 15
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

class_names = ['non-asshole', 'asshole']

# cdf = pd.read_csv("../preprocess_aita_data/super_small_comments_frame.tsv", sep='\t', lineterminator='\n')
cdf = pd.read_csv("../preprocess_aita_data/all_comments_final.tsv", sep='\t', lineterminator='\n')
cdf['labels'] = cdf["asshole_judgement"].apply(get_label)
t = cdf.labels

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')


MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3

all_acc = []
all_prec = []
all_recall = []
all_f1 = []

skf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
cdf = cdf.sample(frac=1).reset_index(drop=True)

for train_index, test_index in skf.split(np.zeros(len(t)), t):

    model = AitaClassifier(len(class_names))
    model = model.to(device)

    df_train = cdf.iloc[train_index]
    df_test = cdf.iloc[test_index]

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    total_steps = len(train_data_loader) * EPOCHS

    optimzer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    loss_fn = nn.CrossEntropyLoss().to(device)

    scheduler = get_linear_schedule_with_warmup(optimzer, num_warmup_steps=0, num_training_steps=total_steps)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimzer, device, scheduler, len(df_train))
        print(f'Train loss {train_loss} accuracy {train_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        if train_acc > best_accuracy:
            # torch.save(model.state_dict(), 'cross_fold_bert.bin')
            best_accuracy = train_acc

    y_comment_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    acc, prec, recall, f1 = get_binary_evaluation_metrics(y_test, y_pred, sig_figs=5)

    all_acc.append(acc)
    all_prec.append(prec)
    all_recall.append(recall)
    all_f1.append(f1)

    # delete the current model on the GPU to ensure cross validation works
    del model


    # cm = confusion_matrix(y_test, y_pred)
    # df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    # show_confusion_matrix(df_cm)


avg_acur = round(np.mean(all_acc), 5)
std_acur = round(np.std(all_acc), 5)

avg_prec = round(np.mean(all_prec), 5)
std_prec = round(np.std(all_prec), 5)

avg_recall = round(np.mean(all_recall), 5)
std_recall = round(np.std(all_recall), 5)

avg_f1 = round(np.mean(all_f1), 5)
std_f1 = round(np.std(all_f1), 5)

print(f'Accuracy: {avg_acur} pm {std_acur}')
print(f'Precision: {avg_prec} pm {std_prec}')
print(f'Recall: {avg_recall} pm {std_recall}')
print(f'F1: {avg_f1} pm {std_f1}')

with open("bert_results", "w+") as b_file:
    write_binary_evaluation_output("Neural BERT", avg_acur, std_acur, avg_prec, std_prec, avg_recall, std_recall, avg_f1, std_f1, b_file)
