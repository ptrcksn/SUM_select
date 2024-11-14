#!/usr/bin/env python
# coding: utf-8

# import necessary libraries:
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset ##### use `DataLoader` and `TensorDataset` to implement batched training

# Simple multi-headed transformer class without feed-forward layers or dropout (based on your code structure)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, input, return_attn=False):
        attn_output, attn_scores = self.attn(input, input, input, need_weights=True, average_attn_weights=False) ##### `average_attn_weights` parameter needs to be set when attention layer is called in forward method, instead of when it is built in constructor
        output = self.layer_norm(input + attn_output)
        if return_attn:
            return [output, attn_scores]
        else:
            return output

# Position-encoded embedding class (tokens and positions are learned during training)
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len, alphabet_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=alphabet_size, embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings=max_len, embedding_dim=embed_dim)
    def forward(self, input):
        positions = torch.arange(0, input.size(1), device=input.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        input = self.token_emb(input)
        embedding = input + positions
        return embedding

# Model definition
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(max_len, alphabet_size, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Mimics GlobalAveragePooling1D
        self.preoutput_layer = nn.Linear(embed_dim, 1)
        self.output_layer = nn.ReLU() ##### our model has final ReLU activation (see below `forward` method)
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.layer_norm(x)
        x = self.transformer_block(x)
        x = x.permute(0, 2, 1)  # Change dimensions for pooling
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.preoutput_layer(x) #####
        x = self.output_layer(x) #####
        return x

# Scheduler function
def set_schedule(epoch): ##### changed name of function to avoid confusion with `scheduler` on line 101
    if epoch < 10:
        return 1.0
    else:
        return float(torch.exp(torch.tensor(-0.9)))

# One-hot encoding function
def one_hot_mine(seq, alphabet=["A", "C", "G", "T"]):
    nums = dict(zip(alphabet + [" "], list(range(1, (len(alphabet) + 1))) + [""]))
    return [nums[s] for s in seq if s != " "]

# Parameters
alphabet = ["A", "C", "G", "T"]
alphabet_size = 10
max_len = 10
embed_dim = 10
num_heads = 2
ff_dim = 10

# Read and preprocess data
with open("compartments_10", "r") as infile:
    dat = [ l.strip().split("\t") for l in infile.readlines() if float(l.strip().split("\t")[4]) <= 1 ]

random.shuffle(dat)
mers = [" ".join([b for b in d[0]]) for d in dat]
encoded_mers = [one_hot_mine(m) for m in mers]
encoded_mers = [e + [0] * (max_len - len(e)) for e in encoded_mers]  # Pad manually since pad_sequences is not available
encoded_mers = torch.tensor(encoded_mers)
labels = torch.tensor([float(d[4]) for d in dat])

# Training and validation split
x_train, y_train = encoded_mers[:len(dat) // 2], labels[:len(dat) // 2]
x_val, y_val = encoded_mers[len(dat) // 2:], labels[len(dat) // 2:]

# Initialize model, optimizer, loss, and scheduler
model = TransformerModel()
optimizer = optim.AdamW(model.parameters())
loss_fn = nn.L1Loss()
scheduler = LambdaLR(optimizer, lr_lambda=set_schedule) #####

# Training loop
num_epochs = 15
batch_size = 128

train_dataset = TensorDataset(x_train, y_train) ##### training data
train_loader = DataLoader(train_dataset, batch_size=batch_size) #####

val_dataset = TensorDataset(x_val, y_val) ##### validation data
val_loader = DataLoader(val_dataset, batch_size=batch_size) #####

##### updated batched training loop:
for epoch in range(num_epochs):
    print('epoch {}:'.format(epoch))
    model.train(True)
    running_loss = 0.
    last_loss = 0.
    for i, train_data in enumerate(train_loader):
        train_inputs, train_labels = train_data
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        loss = loss_fn(train_outputs.squeeze(), train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  training batch {} loss: {}'.format(i+1, last_loss))
            running_loss = 0.
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_inputs, val_labels = val_data
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs.squeeze(), val_labels)
            running_val_loss += val_loss
    mean_val_loss = running_val_loss / i
    print('    mean loss per validation batch: {}'.format(mean_val_loss))

# Prediction
model.eval()
with torch.no_grad():
    results = model(encoded_mers).squeeze().numpy()

# Save predictions
with open("results.tsv", "w") as outfile:
    for i in range(len(dat)):
        outfile.write(f"{dat[i][0]}\t{labels[i].item()}\t{results[i]}\n")

# Attention extraction for specific examples
mers = [" ".join(["G","G","C","G","C","G","A","A","A","A"]), 
        " ".join(["G","C","G","C","G","A","A","A","T","A"]), 
        " ".join(["A","C","A","G","G","G","G","G","G","G"])]
encoded_mers = [one_hot_mine(m) for m in mers]
encoded_mers = [e + [0] * (max_len - len(e)) for e in encoded_mers]
encoded_mers = torch.tensor(encoded_mers)

# Extract attention scores without indexing errors
model.eval()
with torch.no_grad():
    output, attn_scores = model.transformer_block(model.layer_norm(model.embedding_layer(encoded_mers)), return_attn=True) ##### need to normalise embedded input before propagating through `transformer_block` layer
    for m in range(len(mers)):
        for n in range(num_heads):
            with open(f"attn_{m+1}{n+1}.tsv", "w") as outfile:
                for i in range(max_len):
                    # Convert each element to a float using .item() for 0-dim tensors
                    row = [str(attn_scores[m][n][i][j].item()) for j in range(max_len)]
                    outfile.write("\t".join(row) + "\n")
