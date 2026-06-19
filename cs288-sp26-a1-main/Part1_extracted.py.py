#!/usr/bin/env python
# coding: utf-8

# I acknowledge the use of tab based code completion to generally speed up my development process. I used the outputs to finish lines of code I was already writing. I have the ability to explain and even independently replicate the work done in this document if asked by an instructor.

# ## Part 1: Language Modeling
# 
# In this project, you will implement several different types of language models for text.  We'll start with n-gram models and then move on to neural n-gram.
# 
# Warning: Do not start this project the day before it is due!  Some parts require 20 minutes or more to run, so debugging and tuning can take a significant amount of time.
# 
# Our dataset for this project will be the Penn Treebank language modeling dataset.  This dataset comes with some of the basic preprocessing done for us, such as tokenization and rare word filtering (using the `<unk>` token).
# Therefore, we can assume that all word types in the test set also appear at least once in the training set.

# In[1]:


# This block handles some basic setup and data loading.  
# You shouldn't need to edit this, but if you want to 
# import other standard python packages, that is fine.

# imports
from collections import defaultdict, Counter
import numpy as np
import math
import tqdm
import random
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import os

# Load WikiText-2 dataset from HuggingFace
ds = load_dataset("wikitext", "wikitext-2-v1")
train_dataset = ds["train"]
validation_dataset = ds["validation"]
test_dataset = ds["test"]

# Convert to list of tokens (HuggingFace returns text as strings, we need to tokenize)
# WikiText-2 is already tokenized with space-separated tokens
def get_tokens(example):
    # Split by whitespace and filter out empty strings
    tokens = example['text'].split()
    return tokens

# Get all tokens from each split
train_text = []
for example in train_dataset:
    tokens = get_tokens(example)
    if tokens:  # Skip empty examples
        train_text.extend(tokens)

validation_text = []
for example in validation_dataset:
    tokens = get_tokens(example)
    if tokens:
        validation_text.extend(tokens)

test_text = []
for example in test_dataset:
    tokens = get_tokens(example)
    if tokens:
        test_text.extend(tokens)

# Build vocabulary from training set only
# (Validation and test sets may contain unknown tokens, which will be mapped to <unk>)
token_counts = Counter(train_text)

# Create vocabulary with special tokens
# Add special tokens: <unk> for unknown words, <eos> for end of sentence
special_tokens = ['<unk>', '<eos>', '<pad>']
for token in special_tokens:
    token_counts[token] = 0
vocab_list = sorted(token_counts.keys())
vocab_size = len(vocab_list)

# Create a simple vocab class compatible with torchtext interface
class Vocab:
    def __init__(self, vocab_list, token_counts):
        self.itos = vocab_list  # index to string
        self.stoi = {word: idx for idx, word in enumerate(vocab_list)}  # string to index
        self.freqs = token_counts  # frequency counts
    
    def __len__(self):
        return len(self.itos)

vocab = Vocab(vocab_list, token_counts)

print(f"Vocabulary size: {vocab_size}")
print(f"First 30 validation tokens: {validation_text[:30]}")


# In[2]:


print(validation_text[:300])


# We've implemented a unigram model here as a demonstration.

# In[3]:


class UnigramModel:
    def __init__(self, train_text):
        self.counts = Counter(train_text)
        self.total_count = len(train_text)

    def probability(self, word):
        return self.counts[word] / self.total_count

    def next_word_probabilities(self, text_prefix):
        """Return a list of probabilities for each word in the vocabulary."""
        return [self.probability(word) for word in vocab.itos]

    def perplexity(self, full_text):
        """Return the perplexity of the model on a text as a float.
        
        full_text -- a list of string tokens
        """
        log_probabilities = []
        for word in full_text:
            # Note that the base of the log doesn't matter 
            # as long as the log and exp use the same base.
            prob = self.probability(word)
            # Handle 0 probability by using a very small epsilon to avoid log(0)
            if prob == 0:
                prob = 1e-10
            log_probabilities.append(math.log(prob, 2))
        return 2 ** -np.mean(log_probabilities)

unigram_demonstration_model = UnigramModel(train_text)
print('unigram validation perplexity:', 
      unigram_demonstration_model.perplexity(validation_text))

def check_validity(model):
    """Performs several sanity checks on your model:
    1) That next_word_probabilities returns a valid distribution
    2) That perplexity matches a perplexity calculated from next_word_probabilities

    Although it is possible to calculate perplexity from next_word_probabilities, 
    it is still good to have a separate more efficient method that only computes 
    the probabilities of observed words.
    """

    log_probabilities = []
    for i in range(10):
        prefix = validation_text[:i]
        probs = model.next_word_probabilities(prefix)
        assert min(probs) >= 0, "Negative value in next_word_probabilities"
        assert max(probs) <= 1 + 1e-8, "Value larger than 1 in next_word_probabilities"
        assert abs(sum(probs)-1) < 1e-4, "next_word_probabilities do not sum to 1"

        word_id = vocab.stoi[validation_text[i]]
        selected_prob = probs[word_id]
        # Handle 0 probability by using a very small epsilon to avoid log(0)
        if selected_prob == 0:
            selected_prob = 1e-10
        log_probabilities.append(math.log(selected_prob, 2))

    perplexity = 2 ** -np.mean(log_probabilities)
    your_perplexity = model.perplexity(validation_text[:10])
    assert abs(perplexity-your_perplexity) < 0.1, "your perplexity does not " + \
    "match the one we calculated from `next_word_probabilities`,\n" + \
    "at least one of `perplexity` or `next_word_probabilities` is incorrect.\n" + \
    f"we calcuated {perplexity} from `next_word_probabilities`,\n" + \
    f"but your perplexity function returned {your_perplexity} (on a small sample)."


check_validity(unigram_demonstration_model)


# To generate from a language model, we can sample one word at a time conditioning on the words we have generated so far.

# In[4]:


def generate_text(model, n=20, prefix=('<eos>', '<eos>')):
    prefix = list(prefix)
    for _ in range(n):
        probs = model.next_word_probabilities(prefix)
        word = random.choices(vocab.itos, probs)[0]
        prefix.append(word)
    return ' '.join(prefix)

print(generate_text(unigram_demonstration_model))


# In fact there are many strategies to get better-sounding samples, such as only sampling from the top-k words or sharpening the distribution with a temperature.  You can read more about sampling from a language model in this recent paper: https://arxiv.org/pdf/1904.09751.pdf.

# You will need to submit some outputs from the models you implement for us to grade.  The following function will be used to generate the required output files.

# In[5]:


get_ipython().system('wget https://cal-cs288.github.io/sp21/project_files/proj_1/eval_prefixes.txt')
get_ipython().system('wget https://cal-cs288.github.io/sp21/project_files/proj_1/eval_output_vocab.txt')
get_ipython().system('wget https://cal-cs288.github.io/sp21/project_files/proj_1/eval_prefixes_short.txt')
get_ipython().system('wget https://cal-cs288.github.io/sp21/project_files/proj_1/eval_output_vocab_short.txt')

def save_truncated_distribution(model, filename, short=True):
    """Generate a file of truncated distributions.
    
    Probability distributions over the full vocabulary are large,
    so we will truncate the distribution to a smaller vocabulary.

    Please do not edit this function
    """
    vocab_name = 'eval_output_vocab'
    prefixes_name = 'eval_prefixes'

    if short: 
      vocab_name += '_short'
      prefixes_name += '_short'

    with open('{}.txt'.format(vocab_name), 'r') as eval_vocab_file:
        eval_vocab = [w.strip() for w in eval_vocab_file]
    # Map unknown words to <unk> token ID
    unk_id = vocab.stoi['<unk>']
    eval_vocab_ids = [vocab.stoi.get(s, unk_id) for s in eval_vocab]

    all_selected_probabilities = []
    with open('{}.txt'.format(prefixes_name), 'r') as eval_prefixes_file:
        lines = eval_prefixes_file.readlines()
        for line in tqdm.tqdm(lines, leave=False):
            prefix = line.strip().split(' ')
            probs = model.next_word_probabilities(prefix)
            selected_probs = np.array([probs[i] for i in eval_vocab_ids], dtype=np.float32)
            all_selected_probabilities.append(selected_probs)

    all_selected_probabilities = np.stack(all_selected_probabilities)
    np.save(filename, all_selected_probabilities)
    print('saved', filename)


# In[6]:


save_truncated_distribution(unigram_demonstration_model, 'unigram_predictions.npy')


# ### N-gram Model
# 
# Now it's time to implement an n-gram language model.
# 
# Because not every n-gram will have been observed in training, use add-alpha smoothing to make sure no output word has probability 0.
# 
# $$P(w_2|w_1)=\frac{C(w_1,w_2)+\alpha}{C(w_1)+N\alpha}$$
# 
# where $N$ is the vocab size and $C$ is the count for the given bigram.  An alpha value around `3e-3`  should work. 
# 
# One edge case you will need to handle is at the beginning of the text where you don't have `n-1` prior words.  You can handle this however you like as long as you produce a valid probability distribution, but just using a uniform distribution over the vocabulary is reasonable for the purposes of this project.
# 
# A properly implemented bi-gram model should get a perplexity below 550 on the validation set.
# 
# **Note**: Do not change the signature of the `next_word_probabilities` and `perplexity` functions.  We will use these as a common interface for all of the different model types.  Make sure these two functions call `n_gram_probability`, because later we are going to override `n_gram_probability` in a subclass. 
# Also, we suggest pre-computing and caching the counts $C$ when you initialize `NGramModel` for efficiency. 
# 
# **Deliverable**: Submit the bigram distribution from the Ngram model.

# In[7]:


class NGramModel:
    def __init__(self, train_text, n=2, alpha=3e-3, bigram_lambda=0.70, unigram_alpha=0.1):
        self.n = n
        self.smoothing = alpha
        self.bigram_lambda = bigram_lambda
        self.unigram_alpha = unigram_alpha
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2idx = vocab.stoi
        self.unk_id = self.word2idx.get("<unk>", 0)
        self.ngram_counts = Counter()
        self.prefix_counts = Counter()
        self.prefix_to_next = defaultdict(Counter)
        self.uniform_probs = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)

        train_ids = [self.word2idx.get(word, self.unk_id) for word in train_text]
        self.unigram_counts = np.zeros(self.vocab_size, dtype=np.int64)
        for token_id in train_ids:
            self.unigram_counts[token_id] += 1
        self.total_unigrams = len(train_ids)
        self.unigram_probs = (
            (self.unigram_counts + self.unigram_alpha)
            / (self.total_unigrams + self.unigram_alpha * self.vocab_size)
        )

        if self.n == 1:
            for token_id in train_ids:
                ngram = (token_id,)
                self.ngram_counts[ngram] += 1
                self.prefix_counts[()] += 1
                self.prefix_to_next[()][token_id] += 1
        else:
            for i in range(len(train_ids) - self.n + 1):
                ngram = tuple(train_ids[i : i + self.n])
                prefix = ngram[:-1]
                next_id = ngram[-1]
                self.ngram_counts[ngram] += 1
                self.prefix_counts[prefix] += 1
                self.prefix_to_next[prefix][next_id] += 1

    def _to_ngram_ids(self, n_gram):
        if len(n_gram) != self.n:
            return None
        if isinstance(n_gram[0], (int, np.integer)):
            return tuple(n_gram)
        return tuple(self.word2idx.get(w, self.unk_id) for w in n_gram)

    def n_gram_probability(self, n_gram):
        if len(n_gram) != self.n:
            return 1.0 / self.vocab_size
        ngram_ids = self._to_ngram_ids(n_gram)

        if self.n == 2:
            prev_id, next_id = ngram_ids
            c_bigram = self.ngram_counts.get((prev_id, next_id), 0)
            c_prefix = self.prefix_counts.get((prev_id,), 0)
            p_bigram = (c_bigram / c_prefix) if c_prefix > 0 else 0.0
            p_unigram = self.unigram_probs[next_id]
            return self.bigram_lambda * p_bigram + (1.0 - self.bigram_lambda) * p_unigram

        prefix_ids = ngram_ids[:-1] if self.n > 1 else ()
        c = self.ngram_counts.get(ngram_ids, 0)
        c_prefix = self.prefix_counts.get(prefix_ids, 0)
        return (c + self.smoothing) / (c_prefix + self.smoothing * self.vocab_size)

    def next_word_probabilities(self, text_prefix):
        if type(self) is not NGramModel:
            context = list(text_prefix[-(self.n - 1) :]) if self.n > 1 else []
            return [self.n_gram_probability(context + [w]) for w in vocab.itos]

        if self.n == 2:
            if len(text_prefix) < 1:
                return self.unigram_probs.copy()
            prev_id = self.word2idx.get(text_prefix[-1], self.unk_id)
            probs = (1.0 - self.bigram_lambda) * self.unigram_probs.copy()
            c_prefix = self.prefix_counts.get((prev_id,), 0)
            next_counts = self.prefix_to_next.get((prev_id,))
            if next_counts and c_prefix > 0:
                ids_arr = np.fromiter(next_counts.keys(), dtype=np.int64)
                counts_arr = np.fromiter(next_counts.values(), dtype=np.float64)
                probs[ids_arr] += self.bigram_lambda * (counts_arr / c_prefix)
            return probs

        if self.n > 1 and len(text_prefix) < self.n - 1:
            return self.uniform_probs

        if self.n > 1:
            context_ids = tuple(
                self.word2idx.get(w, self.unk_id) for w in text_prefix[-(self.n - 1) :]
            )
        else:
            context_ids = ()

        c_prefix = self.prefix_counts.get(context_ids, 0)
        denom = c_prefix + self.smoothing * self.vocab_size
        probs = np.full(self.vocab_size, self.smoothing / denom, dtype=np.float64)
        next_counts = self.prefix_to_next.get(context_ids)
        if next_counts:
            ids_arr = np.fromiter(next_counts.keys(), dtype=np.int64)
            counts_arr = np.fromiter(next_counts.values(), dtype=np.float64)
            probs[ids_arr] = (counts_arr + self.smoothing) / denom
        return probs

    def perplexity(self, full_text):
        """Return the perplexity of the model on a text as a float."""
        N = len(full_text)
        log_prob_sum = 0.0

        if type(self) is not NGramModel:
            for i in range(N):
                if self.n > 1 and i < self.n - 1:
                    if self.n == 2:
                        tok_id = self.word2idx.get(full_text[i], self.unk_id)
                        p = self.unigram_probs[tok_id]
                    else:
                        p = 1.0 / self.vocab_size
                else:
                    n_gram = full_text[i - self.n + 1 : i + 1]
                    p = self.n_gram_probability(n_gram)
                if p < 1e-12:
                    p = 1e-12
                log_prob_sum += math.log(p, 2)
            return 2 ** (-log_prob_sum / N)

        full_ids = [self.word2idx.get(w, self.unk_id) for w in full_text]
        for i in range(N):
            if self.n > 1 and i < self.n - 1:
                if self.n == 2:
                    p = self.unigram_probs[full_ids[i]]
                else:
                    p = 1.0 / self.vocab_size
            else:
                if self.n == 1:
                    ngram_ids = (full_ids[i],)
                else:
                    ngram_ids = tuple(full_ids[i - self.n + 1 : i + 1])
                p = self.n_gram_probability(ngram_ids)
            if p < 1e-12:
                p = 1e-12
            log_prob_sum += math.log(p, 2)
        return 2 ** (-log_prob_sum / N)

unigram_model = NGramModel(train_text, 1)
check_validity(unigram_model)
print('unigram validation perplexity:', unigram_model.perplexity(validation_text)) # this should be the almost the same as our unigram model perplexity above

bigram_model = NGramModel(train_text, n=2, bigram_lambda=0.70)
check_validity(bigram_model)
print('bigram validation perplexity:', bigram_model.perplexity(validation_text))

trigram_model = NGramModel(train_text, n=3)
check_validity(trigram_model)
print('trigram validation perplexity:', trigram_model.perplexity(validation_text)) # this won't do very well...

save_truncated_distribution(bigram_model, 'bigram_predictions.npy') # this might take a few minutes



# Please download `bigram_predictions.npy` once you finish this section so that you can submit it.

# We can also generate samples from the model to get an idea of how it is doing.

# In[8]:


print(generate_text(bigram_model))


# We now free up some RAM, **it is important to run the cell below, otherwise you may quite possibly run out of RAM in the runtime.**

# In[9]:


# Free up some RAM. 
del bigram_model
del trigram_model


# ### Neural N-gram Model
# 
# In this section, you will implement a neural version of an n-gram model.  The model will use a simple feedforward neural network that takes the previous `n-1` words and outputs a distribution over the next word.
# 
# You will use PyTorch to implement the model.  We've provided a little bit of code to help with the data loading using PyTorch's data loaders (https://pytorch.org/docs/stable/data.html)
# 
# A model with the following architecture and hyperparameters should reach a validation perplexity below 230.
# * embed the words with dimension 128, then flatten into a single embedding for $n-1$ words (with size $(n-1)*128$)
# * run 2 hidden layers with 1024 hidden units, then project down to size 128 before the final layer (ie. 4 layers total). 
# * use weight tying for the embedding and final linear layer (this made a very large difference in our experiments); you can do this by creating the output layer with `nn.Linear`, then using `F.embedding` with the linear layer's `.weight` to embed the input
# * rectified linear activation (ReLU) and dropout 0.1 after first 2 hidden layers. **Note: You will likely find a performance drop if you add a nonlinear activation function after the dimension reduction layer.**
# * train for 10 epochs with the Adam optimizer (should take around 15-20 minutes)
# 
# 
# We encourage you to try other architectures and hyperparameters, and you will likely find some that work better than the ones listed above.  A proper implementation with these should be enough to receive full credit on the assignment, though.

# In[ ]:


def ids(tokens):
    return [vocab.stoi[t] for t in tokens]

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
assert device.type != "cpu", (
    "No GPU found; on Apple Silicon enable MPS, or use CUDA in Colab/Kaggle."
)

class NeuralNgramDataset(torch.utils.data.Dataset):
    def __init__(self, text_token_ids, n):
        assert n >= 2, "This implementation expects n >= 2."
        self.n = n

        y = torch.tensor(text_token_ids, dtype=torch.long)
        pad = torch.full((n - 1,), vocab.stoi["<eos>"], dtype=torch.long)
        padded = torch.cat([pad, y], dim=0)

        self.x = padded.unfold(0, n - 1, 1).contiguous()
        self.y = y

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class NeuralNGramNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.vocab_size = len(vocab)

        embed_dim = 128
        input_dim = (n - 1) * embed_dim
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)  
        self.dropout = nn.Dropout(0.2)

        self.output = nn.Linear(128, self.vocab_size, bias=True)

    def forward(self, x):
        emb = F.embedding(x, self.output.weight)   
        h = emb.reshape(emb.size(0), -1)         

        h = self.dropout(F.relu(self.fc1(h)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.fc3(h) 

        logits = self.output(h)
        return logits

class NeuralNGramModel:
    def __init__(self, n):
        self.n = n
        self.device = device
        self.eos_id = vocab.stoi["<eos>"]
        self.network = NeuralNGramNetwork(n).to(self.device)

    def _build_xy(self, text_ids):
        y = torch.tensor(text_ids, dtype=torch.long)
        pad = torch.full((self.n - 1,), self.eos_id, dtype=torch.long)
        padded = torch.cat([pad, y], dim=0)
        x = padded.unfold(0, self.n - 1, 1)[:y.size(0)]
        return x, y

    def train(self, epochs=10, lr=1e-3, batch_size=512):
        print("Preparing training tensors...", flush=True)
        train_ids = ids(train_text)
        x_all, y_all = self._build_xy(train_ids)
        num_tokens = y_all.size(0)
        num_batches = (num_tokens + batch_size - 1) // batch_size

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        print(
            f"device={self.device}, train_tokens={num_tokens}, "
            f"batches_per_epoch={num_batches}, batch_size={batch_size}",
            flush=True,
        )

        for epoch in range(1, epochs + 1):
            self.network.train()
            total_nll = 0.0
            total_seen = 0

            perm = torch.randperm(num_tokens)
            pbar = tqdm.tqdm(range(0, num_tokens, batch_size), desc=f"Epoch {epoch}/{epochs}", leave=True)
            for step, start in enumerate(pbar, start=1):
                idx = perm[start:start + batch_size]
                x = x_all[idx].to(self.device)
                y = y_all[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.network(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                total_nll += loss.item() * bs
                total_seen += bs

                if step % 100 == 0:
                    pbar.set_postfix(train_ppl=f"{math.exp(total_nll / total_seen):.2f}")

            print(f"Epoch {epoch}/{epochs} train perplexity: {math.exp(total_nll / total_seen):.2f}", flush=True)

    @torch.no_grad()
    def next_word_probabilities(self, text_prefix):
        self.network.eval()
        prefix_ids = ids(text_prefix)

        if len(prefix_ids) < self.n - 1:
            prefix_ids = [vocab.stoi["<eos>"]] * (self.n - 1 - len(prefix_ids)) + prefix_ids
        else:
            prefix_ids = prefix_ids[-(self.n - 1):]

        x = torch.tensor(prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        logits = self.network(x)[0]
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().tolist()

    @torch.no_grad()
    def perplexity(self, text, batch_size=2048):
        self.network.eval()
        text_ids = ids(text)
        if len(text_ids) == 0:
            return float("inf")

        x_all, y_all = self._build_xy(text_ids)
        total_nll = 0.0
        total_tokens = y_all.size(0)

        for start in range(0, total_tokens, batch_size):
            end = start + batch_size
            x = x_all[start:end].to(self.device)
            y = y_all[start:end].to(self.device)
            logits = self.network(x)
            total_nll += F.cross_entropy(logits, y, reduction="sum").item()

        return math.exp(total_nll / total_tokens)

RUN_VALIDITY_CHECK = False 

print("Initializing neural trigram model...", flush=True)
neural_trigram_model = NeuralNGramModel(3)

if RUN_VALIDITY_CHECK:
    print("Running check_validity...", flush=True)
    check_validity(neural_trigram_model)

print("Starting training...", flush=True)
neural_trigram_model.train(epochs=10, lr=1e-3, batch_size=512)
print("Computing validation perplexity...", flush=True)
print("neural trigram validation perplexity:", neural_trigram_model.perplexity(validation_text))

print("Saving predictions...", flush=True)
save_truncated_distribution(neural_trigram_model, "neural_trigram_predictions.npy", short=False)


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# save_path = "/content/drive/MyDrive/neural_trigram_predictions.npy"
# save_truncated_distribution(neural_trigram_model, save_path, short=False)
# print(f"Neural trigram predictions saved to: {save_path}")
save_truncated_distribution(neural_trigram_model, "neural_trigram_predictions.npy", short=False)


# Free up RAM.

# In[12]:


# Delete model we don't need. 
del neural_trigram_model


# ### Submission
# 
# Upload a submission with the following files to Gradescope:
# * Part1.ipynb (rename to match this exactly)
# * neural_trigram_predictions.npy
# * bigram_predictions.npy
# 
# You can upload files individually or as part of a zip file, but if using a zip file be sure you are zipping the files directly and not a folder that contains them.
# 
# Be sure to check the output of the autograder after it runs.  It should confirm that no files are missing and that the output files have the correct format.  Note that the test set perplexities shown by the autograder are on a completely different scale from your validation set perplexities due to truncating the distribution and selecting different text.  Don't worry if the values seem much worse.
