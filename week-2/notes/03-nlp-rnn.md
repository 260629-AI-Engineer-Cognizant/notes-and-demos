# NLP, RNNs, Embeddings, Transformers, and TensorBoard

# Why NLP Is Different

Images are already numeric once loaded into a computer.

An image is usually represented as pixel values:

```text
height x width x color channels
```

Text is different. A sentence like this:

```text
This movie was surprisingly good.
```

is not naturally numeric.

Before a neural network can use text, we need to convert it into numbers while preserving useful meaning.

A beginner NLP pipeline often looks like this:

```text
raw text
-> tokens
-> integer IDs
-> padded sequences
-> embeddings
-> model
-> prediction
```

A modern pretrained Transformer pipeline often looks like this:

```text
raw text
-> tokenizer/preprocessor
-> token IDs + masks
-> pretrained Transformer
-> classifier head
-> prediction
```

Bite-size version:

> NLP models need text to be converted into numeric representations before they can learn from it.

Interview version:

> Text data must be transformed into numeric representations before it can be used by a neural network. Common NLP preprocessing steps include tokenization, vocabulary mapping, padding or truncation, and embedding. Modern Transformer models also use token IDs, attention masks, and pretrained language representations.

---

# Tokenization

## What is a token?

A **token** is a piece of text.

For beginner examples, a token is often a word.

Example:

```text
"this movie was good"
```

could become:

```text
["this", "movie", "was", "good"]
```

Then each token gets mapped to an integer ID:

```text
["this", "movie", "was", "good"]
-> [11, 47, 18, 92]
```

Each integer represents a token in the model's vocabulary.

## Word tokens vs subword tokens

Older beginner examples often use word-level tokenization.

Modern Transformer models usually use **subword tokenization**.

Example idea:

```text
"unbelievable"
-> ["un", "believ", "able"]
```

This helps models handle:

- rare words
- long words
- misspellings
- words not seen exactly during training
- shared word parts across related words

Example:

```text
train
training
trained
trainer
```

These words share meaning. Subword tokenization gives the model a better chance to reuse pieces of language.

Bite-size version:

> Tokenization breaks text into smaller pieces that a model can map to numbers.

Interview version:

> Tokenization is the process of splitting raw text into units such as words, characters, or subword pieces. These tokens are then mapped to integer IDs so they can be passed into a neural network. Modern Transformers usually use subword tokenization because it handles rare and unseen words better than simple word-level tokenization.

---

# Vocabulary and Integer IDs

A **vocabulary** is the set of tokens the model knows how to represent.

Example vocabulary:

```text
0 -> padding
1 -> unknown
2 -> this
3 -> movie
4 -> good
5 -> bad
```

The sentence:

```text
"this movie good"
```

could become:

```text
[2, 3, 4]
```

The numbers themselves do **not** have meaningful numeric relationships.

For example, if:

```text
4 -> good
5 -> bad
```

that does not mean:

```text
bad is mathematically greater than good
```

The integer IDs are just lookup keys.

The model needs an embedding layer to turn those IDs into useful vectors.

Bite-size version:

> Token IDs are not meaningful numbers by themselves. They are lookup keys.

Interview version:

> After tokenization, tokens are usually mapped to integer IDs from a fixed vocabulary. These IDs identify tokens, but they should not be treated as continuous numeric features. An embedding layer or pretrained model converts those IDs into dense numeric vectors that can represent useful relationships between tokens.

---

# Padding, Truncation, and Masks

Text examples can have different lengths.

Example:

```text
"good movie"
"this movie was surprisingly good and very funny"
```

Neural networks train in batches, and batches need consistent shapes.

So we often make sequences the same length.

## Padding

Short sequences get extra values added.

```text
[12, 45, 90]
-> [12, 45, 90, 0, 0, 0]
```

## Truncation

Long sequences are cut down to a maximum length.

```text
[4, 8, 15, 16, 23, 42, 99, 100]
-> [4, 8, 15, 16]
```

## Attention masks

Transformer models often use an **attention mask** to distinguish real tokens from padding.

Conceptually:

```text
token IDs:       [12, 45, 90, 0, 0, 0]
attention mask:  [ 1,  1,  1, 0, 0, 0]
```

The mask tells the model:

```text
1 -> pay attention to this token
0 -> ignore this padding
```

Bite-size version:

> Padding makes batches the same shape. Masks tell the model which tokens are real.

Interview version:

> Padding is used to make variable-length text sequences fit into fixed-size batches. Truncation limits sequences that are too long. Transformer models often use attention masks so the model can ignore padding tokens during attention calculations.

---

# Embeddings

## What is an embedding?

An **embedding** is a dense vector representation of a token.

Instead of representing a word as one integer, the model represents it as a vector of numbers.

Example:

```text
"good" -> [0.25, -0.81, 0.10, 0.44]
```

A Keras embedding layer might look like this:

```python
tf.keras.layers.Embedding(
    input_dim=10000,
    output_dim=64
)
```

This means:

```text
input_dim=10000
The model supports 10,000 possible token IDs.

output_dim=64
Each token becomes a vector of 64 numbers.
```

So the embedding table has this shape:

```text
10000 tokens x 64 values per token
```

That means the embedding layer has:

```text
10000 * 64 = 640,000 trainable parameters
```

## Why embeddings matter

Embeddings let models learn relationships between words.

For a sentiment model, words like these may learn similar representations:

```text
great, excellent, amazing
```

And these may learn different but related representations:

```text
bad, boring, awful
```

The model is not told these relationships manually. It learns them from data.

## Shape example

Suppose one review has 200 tokens, and each token gets a 64-number embedding.

Before embedding:

```text
(200,)
```

After embedding:

```text
(200, 64)
```

For a batch of 32 reviews:

```text
(32, 200, 64)
```

That means:

```text
32 reviews
200 tokens per review
64 values per token
```

Bite-size version:

> An embedding turns each token ID into a learned vector.

Interview version:

> An embedding layer maps discrete token IDs into dense, trainable vectors. These vectors allow the model to learn useful relationships between tokens based on the task. Instead of treating words as unrelated integer IDs or huge one-hot vectors, embeddings represent words in a lower-dimensional continuous space.

---

# Why Embeddings Matter for Vector Databases

This is important for Week 3.

A vector database stores and searches vectors.

Tools such as Chroma and Pinecone are commonly used for semantic search and retrieval-augmented generation workflows.

The core idea:

```text
text -> embedding model -> vector -> vector database
```

If two pieces of text have similar meaning, their embeddings should be close together in vector space.

Example:

```text
"How do I reset my password?"
"Can I change my login password?"
```

These sentences use different words, but they mean similar things.

A good embedding model should place them near each other.

Vector databases let us search by meaning instead of exact keywords.

Traditional keyword search asks:

```text
Do these words match?
```

Vector search asks:

```text
Are these meanings close?
```

Bite-size version:

> Embeddings turn text into vectors. Vector databases store those vectors so we can search by meaning.

Interview version:

> Embeddings are dense vector representations of text. In vector databases such as Chroma or Pinecone, documents are converted into embeddings and stored for similarity search. This enables semantic search, where results are retrieved based on meaning rather than exact keyword overlap.

---

# Baseline NLP Model: Embedding + Pooling

A strong beginner baseline is:

```text
TextVectorization
-> Embedding
-> GlobalAveragePooling1D
-> Dense
-> Output
```

## What does pooling do?

After the embedding layer, each token has its own vector.

For example:

```text
200 tokens -> 200 embedding vectors
```

`GlobalAveragePooling1D` averages those token vectors into one review vector.

Conceptually:

```text
all token vectors -> average -> one review vector
```

This model mostly asks:

```text
What words showed up overall?
```

That works well for sentiment classification because many reviews contain obvious sentiment words:

```text
great, excellent, boring, awful, disappointing
```

## Strengths

- fast
- simple
- easy to explain
- often surprisingly accurate
- good baseline for comparison

## Weakness

It does not deeply model word order.

Example:

```text
I thought it would be bad, but it was actually great.
```

A pooling model may see both negative and positive words but may not fully understand the shift in meaning.

Bite-size version:

> Pooling averages word vectors into one text-level vector.

Interview version:

> A simple text classifier can use an embedding layer followed by global pooling and dense layers. Pooling summarizes the sequence by averaging token embeddings, which creates a fixed-size representation for the full text. This is efficient and often strong for sentiment classification, but it does not explicitly model word order.

---

# Recurrent Neural Networks

## What does recurrent mean?

A network is **recurrent** when it reuses the same layer repeatedly across steps in a sequence.

For text, each word or token is a step.

Example:

```text
"I really liked it"
```

A recurrent layer processes it like this:

```text
Step 1: I
Step 2: really
Step 3: liked
Step 4: it
```

At each step, it updates a memory called the hidden state.

Conceptually:

```text
current token + previous memory -> new memory
```

## Basic RNN

A basic RNN reads a sequence one token at a time and carries information forward.

Example:

```text
The -> movie -> was -> not -> good
```

The RNN tries to remember earlier words as it reads later words.

## Problem with basic RNNs

Basic RNNs can struggle with long-term dependencies.

Example:

```text
The movie, despite the strong opening and impressive visuals, was disappointing.
```

The important relationship is:

```text
movie -> was disappointing
```

But many words appear in between.

Basic RNNs often have trouble preserving important information across long sequences.

Bite-size version:

> An RNN reads a sequence step by step while carrying memory forward.

Interview version:

> A recurrent neural network processes sequential data one time step at a time. At each step, it combines the current input with a hidden state from the previous step. This allows it to model order, but basic RNNs often struggle with long-term dependencies due to issues such as vanishing gradients.

---

# GRUs

GRU stands for **Gated Recurrent Unit**.

A GRU is an improved recurrent layer.

It uses gates to decide:

```text
What should I keep?
What should I forget?
What new information matters?
```

A GRU is usually simpler than an LSTM but still much better than a basic RNN for many sequence tasks.

## Why use a GRU?

The GRU model asks:

```text
How does the meaning change as I read this sequence in order?
```

That can help with phrases like:

```text
not good
started strong but ended badly
I expected to hate it, but loved it
```

## Why a GRU may not always beat pooling

A GRU is more complex than pooling.

More complexity can mean:

- more parameters
- slower training
- more chances to overfit
- more sensitivity to learning rate, dropout, and sequence length

For IMDB sentiment classification, a simple pooling model can do very well because many examples are keyword-heavy.

That does not mean the GRU is useless. It means the task may not require much sequence memory.

## How we improve the GRU demo

In the demo, we make the GRU more stable by using:

- full IMDB training data
- an embedding with masking
- dropout
- a bidirectional GRU
- early stopping
- learning-rate reduction when validation loss stops improving

Bite-size version:

> A GRU is a recurrent model with gates that help it remember useful sequence information.

Interview version:

> A GRU is a gated recurrent neural network layer that improves on a basic RNN by controlling how information is updated and retained over time. It has fewer gates than an LSTM, making it simpler and often faster, while still handling sequence dependencies better than a vanilla RNN.

---

# LSTMs

LSTM stands for **Long Short-Term Memory**.

An LSTM is another gated recurrent layer.

It has a more detailed memory system than a GRU.

An LSTM decides:

```text
what to forget
what to keep
what to output
```

LSTMs were widely used for NLP before Transformers became dominant.

## GRU vs LSTM

Simple comparison:

```text
GRU  = simpler gated recurrent model
LSTM = more complex gated recurrent model
```

Both are designed to handle sequence memory better than a basic RNN.

LSTMs may perform better on some tasks, but GRUs are often faster and easier to use.

Bite-size version:

> An LSTM is a more complex gated RNN designed to remember information over longer sequences.

Interview version:

> An LSTM is a recurrent neural network architecture designed to handle long-term dependencies better than a basic RNN. It uses gates and a cell state to control what information is remembered, forgotten, and output. LSTMs were heavily used in NLP and time-series tasks before Transformers became the standard for many language problems.

---

# Transformers

Transformers changed NLP by replacing step-by-step recurrence with **attention**.

## RNN-style thinking

Recurrent models read in order:

```text
word 1 -> memory -> word 2 -> memory -> word 3 -> memory
```

## Transformer-style thinking

Transformers let tokens compare themselves directly to other tokens:

```text
word 1 <-> word 2 <-> word 3 <-> word 4
```

This is useful because meaning often depends on relationships between words that may not be next to each other.

Example:

```text
The movie was not bad at all.
```

The model needs to connect:

```text
not <-> bad
```

A Transformer can learn which tokens should pay attention to each other.

## Why Transformers are strong for NLP

Transformers are commonly used because they:

- handle long-range relationships well
- process sequences more parallelly than RNNs
- scale well with large datasets
- support transfer learning through pretrained models
- power models like BERT-style classifiers and GPT-style generators

Bite-size version:

> A Transformer uses attention so tokens can directly focus on other important tokens.

Interview version:

> A Transformer is a neural network architecture based on attention rather than recurrence. Instead of processing tokens strictly one at a time, attention allows each token to weigh the importance of other tokens in the sequence. This makes Transformers effective for long-range context and highly scalable, which is why pretrained Transformers are widely used in modern NLP.

---

# Transfer Learning with Pretrained Transformers

Training a language model from scratch is expensive.

A pretrained Transformer has already learned language patterns from a large text corpus.

Instead of starting from zero, we reuse that knowledge.

For classification, the common workflow is:

```text
raw text
-> pretrained tokenizer/preprocessor
-> pretrained Transformer backbone
-> classification head
-> prediction
```

## Backbone and classifier head

A pretrained model often has two main parts:

```text
backbone: the main language understanding model
head: the task-specific output layer
```

For sentiment classification:

```text
backbone: understands language patterns
head: predicts negative vs positive
```

## Fine-tuning

Fine-tuning means we continue training the pretrained model on our specific task.

Example:

```text
general English knowledge
-> fine-tune on movie reviews
-> sentiment classifier
```

In this lesson, the pretrained Transformer section is **required**, not optional.

We are not building the Transformer from scratch. We are using it the way engineers usually use it in practice.

Bite-size version:

> Transfer learning means starting with a pretrained model and adapting it to your task.

Interview version:

> In NLP, transfer learning commonly means starting with a pretrained Transformer model and fine-tuning it on a downstream task such as sentiment classification. The Transformer backbone provides general language representations, and a task-specific classification head maps those representations to labels. This is often more effective and efficient than training a language model from scratch.

---

# TensorBoard

TensorBoard is a dashboard for inspecting model training.

Instead of only seeing printed accuracy values, TensorBoard lets us compare training runs visually.

We can monitor:

- training loss
- validation loss
- training accuracy
- validation accuracy
- precision
- recall
- overfitting signs

## Why validation loss matters

Training loss measures performance on the data the model learns from.

Validation loss measures performance on data the model does not train on.

If training loss keeps improving but validation loss gets worse, the model may be overfitting.

Example pattern:

```text
training loss down
validation loss up
```

This usually means:

```text
The model is memorizing training data instead of generalizing.
```

## Minimal Keras callback

```python
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir="logs/model_name"
)
```

Then include it during training:

```python
model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[tensorboard_cb]
)
```

Open TensorBoard in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

Bite-size version:

> TensorBoard is a visual dashboard for training metrics.

Interview version:

> TensorBoard is a visualization tool used to monitor machine learning experiments. It can track metrics such as loss, accuracy, validation performance, and model graphs. It is especially useful for comparing experiments and identifying overfitting or underfitting.

---

# Binary Sentiment Classification

The IMDB dataset is a binary sentiment classification dataset.

Labels:

```text
0 -> negative review
1 -> positive review
```

For a custom binary classifier, the final layer often looks like this:

```python
tf.keras.layers.Dense(1, activation="sigmoid")
```

And the loss is usually:

```python
loss="binary_crossentropy"
```

For a pretrained Transformer classifier, the model may output two logits:

```text
[negative_score, positive_score]
```

Then we often use:

```python
SparseCategoricalCrossentropy(from_logits=True)
```

Bite-size version:

> Binary sentiment classification predicts one of two labels: negative or positive.

Interview version:

> Sentiment classification is a supervised NLP task where the model predicts the emotional polarity of text. In binary sentiment classification, labels usually represent negative and positive sentiment. A custom model might use a sigmoid output with binary crossentropy, while a two-class Transformer classifier may use logits with sparse categorical crossentropy.

---

# Quick Cheat Sheet

| Topic | Bite-size explanation | Interview-level explanation |
|---|---|---|
| Tokenization | Split text into pieces | Converts raw text into units such as words or subwords that can be mapped to IDs |
| Vocabulary | Known token list | Fixed set of tokens the model can represent |
| Padding | Make sequences equal length | Adds placeholder values so batches have consistent shape |
| Attention mask | Ignore padding | Identifies which tokens are real and which are padding |
| Embedding | Token ID to vector | Dense learned vector representation of a token |
| Pooling | Average token vectors | Summarizes a sequence into a fixed-size vector |
| RNN | Reads in order | Processes sequences step by step using a hidden state |
| GRU | Simpler gated RNN | Uses gates to keep or forget information over time |
| LSTM | More complex gated RNN | Uses gates and a cell state to model longer-term dependencies |
| Transformer | Attention-based model | Uses attention so tokens can directly consider other tokens |
| Transfer learning | Reuse pretrained model | Fine-tunes a pretrained model on a specific downstream task |
| TensorBoard | Training dashboard | Visualizes loss, accuracy, precision/recall context, and helps compare model experiments |
| Vector database | Search vectors | Stores embeddings for similarity search and semantic retrieval |