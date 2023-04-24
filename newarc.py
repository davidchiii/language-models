import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler, DataCollatorWithPadding
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from accelerate import Accelerator

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


TOKENIZE = True
if TOKENIZE:
    df = pd.read_csv('data/train.csv')
    tokens = [tokenizer(a, max_length=512, truncation=True)
              for a in tqdm(df.comment_text.values)]
    df['tokens'] = tokens
    df.to_pickle('df.pkl')
else:
    df = pd.read_pickle('df.pkl')


label_cols = list(df.iloc[:, 2:-1].columns.values)
print(df[label_cols].sum(axis=1).max())
print(label_cols)

kf = KFold(n_splits=2)
folds = dict()
for i, (train_index, test_index) in enumerate(kf.split(df)):
    folds[i] = {'train': df.iloc[train_index], 'test': df.iloc[test_index]}

train_datasets = []
eval_datasets = []
models = []
NUM_LABELS = len(label_cols)
for i in folds:
    train_y = folds[i]['train'][label_cols].values
    valid_y = folds[i]['test'][label_cols].values

    train_dataset = folds[i]['train'].tokens.values
    for a, b in zip(train_dataset, train_y):
        a['label'] = b
    train_datasets.append(train_dataset)

    eval_dataset = folds[i]['test'].tokens.values
    for a, b in zip(eval_dataset, valid_y):
        a['label'] = b
    eval_datasets.append(eval_dataset)
    models.append(AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS))

train_datasets = np.array(train_datasets)
eval_datasets = np.array(eval_datasets)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 32
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
num_epochs = 2
for i in range(len(models)):
    AUCs = []
    print('model', i)
    models[i].to(device)
    optimizer = AdamW(models[i].parameters(), lr=1e-4)

    train_dataloader = DataLoader(
        train_datasets[i],
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        eval_datasets[i],
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    accelerator = Accelerator()
    train_dataloader, eval_dataloader, models[i], optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, models[i], optimizer
    )

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_loss_set = []
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        num_training_steps = len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        models[i].train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = models[i](
                batch['input_ids'],
                attention_mask=batch['attention_mask'])
            logits = outputs[0]
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits.view(-1, NUM_LABELS),
                             batch['labels'].type_as(logits).view(-1, NUM_LABELS))
            train_loss_set.append(loss.item())
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            tr_loss += loss.item()
            nb_tr_examples += batch['input_ids'].size(0)
            nb_tr_steps += 1
            progress_bar.update(1)
        print("Train loss: {}".format(tr_loss/nb_tr_steps))

models[1].save_pretrained('export')
