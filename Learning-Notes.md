# 📚 Learning Notes — Master Hub

> My personal, growing knowledge base across **all courses, topics, and self-study**.
> Structured for **future reference** and **interview revision**.
>
> Organized as: **Backlog** (what to learn next) → **Courses** (structured notes per course) → **Standalone Topics**.

**Legend:** 💡 = key intuition · ⚠️ = common gotcha/interview trap · 🎯 = interview soundbite

---

## 🗂️ Master Index

- **[📌 To Learn / Revise — Backlog](#-to-learn--revise--backlog)**
- **Courses**
  - [Course 1 · Karpathy — Let's Build GPT from Scratch](#course-1--karpathy--lets-build-gpt-from-scratch)
- **Standalone Topics**
  - _(none yet — added as I study individual topics outside a course)_

> **How this file grows:** each new course gets its own `## Course N · <title>` section with its own local table of contents. Standalone deep-dives (e.g. tokenizers, optimizers) get a `## Topic · <name>` section. The backlog below tracks what's queued.

---

## 📌 To Learn / Revise — Backlog

> Source: `Things2read.txt`. Move an item into a proper Course/Topic section once studied, and check it off here.

- [ ] **Tokenizers** — character-level vs sub-word vs word-level; pros/cons of each; implementations: **tiktoken**, **SentencePiece**, **BPE** (Byte-Pair Encoding) and how they compare.
- [ ] **Long-context techniques** — **RoPE** (Rotary Position Embedding) & RoPE scaling; **sparse attention**; **ring attention** — how they extend the context window.
- [ ] **Loss functions** — **cross-entropy** and others (MSE, NLL, etc.); when each is used and why.
- [ ] **Activation functions & optimizers** — ReLU/GELU/SiLU etc.; SGD/Adam/AdamW etc.; trade-offs.

_Add new items here as they come up; link to the section once written up._

---

# Course 1 · Karpathy — Let's Build GPT from Scratch

> Notes from Karpathy's "Let's build GPT: from scratch, in code, spelled out" + supporting Q&A.

### Course Table of Contents
1. [File I/O & Text Encoding](#1-file-io--text-encoding)
2. [Python Building Blocks](#2-python-building-blocks)
3. [The Data Pipeline (text → tensors)](#3-the-data-pipeline-text--tensors)
4. [PyTorch Core Concepts](#4-pytorch-core-concepts)
5. [Batching & Sampling](#5-batching--sampling)
6. [Embeddings](#6-embeddings)
7. [The Bigram Model](#7-the-bigram-model)
8. [Embedding Techniques (Landscape)](#8-embedding-techniques-landscape)
9. [Quick Interview Cheat-Sheet](#9-quick-interview-cheat-sheet)

---

## 1. File I/O & Text Encoding

> The notebook's very first step: read the raw text file off disk.

### `open()` & reading files
```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```
- `open(filename, mode, encoding)` returns a **file object** (a handle).
- **Modes:** `"r"` read · `"w"` write (overwrites) · `"a"` append · `"rb"`/`"wb"` binary. Must be a **string** — bare `r` → `NameError`.
- **`with open(...) as f:`** = *context manager* → auto-closes the file when the block ends, even on error. Preferred way.
- `f.read()` reads the whole file into one string.

### Text encoding (`encoding='utf-8'`)
- A file on disk is just **bytes**. An *encoding* is the rule mapping bytes ↔ characters.
- ⚠️ Always specify `encoding='utf-8'` explicitly — the OS default varies (esp. Windows) → can corrupt text or raise `UnicodeDecodeError`.

### What UTF-8 is
- **Unicode** = catalog giving every character a unique number (code point).
- **UTF-8** = scheme to store those numbers as bytes; **variable-width (1–4 bytes)**:
  - ASCII/English → 1 byte
  - Latin accents, Greek, Cyrillic, Arabic/**Urdu** → 2 bytes
  - Chinese/Japanese/Korean → 3 bytes
  - Emoji, rare scripts → 4 bytes
- Won because it's **ASCII-compatible**, compact for English, and self-synchronizing.
- 💡 Urdu is fully supported (Arabic script, ~2 bytes/char). Encoding isn't the issue — display depends on **fonts** + **RTL** rendering (separate concern).

### UTF-8 vs ASCII
- **ASCII** = 128 fixed chars, always 1 byte, English only.
- **UTF-8** = superset of ASCII covering all ~150,000 Unicode chars.
- For plain English, both produce **identical bytes**. ASCII crashes beyond 128 chars; UTF-8 handles everything.

### How UTF-8 knows byte boundaries (the clever part)
Leading bits of each byte act as a header giving the length:

| Byte pattern | Meaning |
|---|---|
| `0xxxxxxx` | 1-byte char (ASCII) |
| `110xxxxx` | start of 2-byte char |
| `1110xxxx` | start of 3-byte char |
| `11110xxx` | start of 4-byte char |
| `10xxxxxx` | **continuation** byte (middle of a char) |

- Number of leading `1`s = how many bytes the char uses. Header bits stripped, rest glued into the code point.
- 💡 Makes it **self-synchronizing** (recover from mid-stream) and **unambiguous**.

### UTF-16 & UTF-32 (why they exist)
- **UTF-16**: 2 or 4 bytes. Legacy of Windows/Java/JavaScript; also **more compact for CJK** (2 bytes vs UTF-8's 3).
- **UTF-32**: always 4 bytes (fixed-width) → **instant Nth-char indexing**, but wastes space. Mostly used **in memory**, not files.
- 🎯 **UTF-8 wins for storage/web/files.** Others survive due to legacy platforms & specific trade-offs.

---

## 2. Python Building Blocks

### `enumerate(iterable, start=0)`
Returns `(index, item)` pairs while looping — no manual counter needed.

```python
for i, ch in enumerate(['a', 'b', 'c']):
    print(i, ch)      # 0 a / 1 b / 2 c

# start index
for i, ch in enumerate(['a', 'b'], start=1):
    print(i, ch)      # 1 a / 2 b
```

**Used in GPT** to build char↔int vocab maps:
```python
stoi = { ch: i for i, ch in enumerate(chars) }   # string -> int
itos = { i: ch for i, ch in enumerate(chars) }   # int -> string
```

---

### `lambda` — anonymous one-line function
`lambda args: expression` — auto-returns the expression. One expression only.

```python
square = lambda x: x * x        # square(5) -> 25
add    = lambda a, b: a + b     # add(3, 4) -> 7
```
💡 Best used **inline** as an argument to `sort`, `map`, `filter`. Use `def` for anything longer.

**Used in GPT:**
```python
encode = lambda s: [stoi[c] for c in s]            # string -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])   # list[int] -> string
```

| Use lambda | Use def |
|---|---|
| short, one expression | multiple statements |
| passed inline (`key=`, `map`) | reused in many places |
| throwaway | needs name/docstring |

---

### `sort` / `sorted`, `map`, `filter`

| Function | Does | Returns |
|---|---|---|
| `sorted(it)` / `.sort()` | reorders items | new list / in-place (`None`) |
| `map(fn, it)` | transforms each item | lazy iterator → wrap in `list()` |
| `filter(fn, it)` | keeps items where `fn` is True | lazy iterator → wrap in `list()` |

```python
sorted([3,1,2])                          # [1,2,3]
sorted(words, key=len)                    # sort by length
sorted(words, key=lambda w: w[-1])        # sort by last letter

list(map(lambda x: x**2, [1,2,3]))        # [1,4,9]
list(filter(lambda x: x%2==0, [1,2,3,4])) # [2,4]
```
⚠️ `map`/`filter` return **lazy iterators**, not lists — wrap in `list()` to see values.
💡 Modern Python often prefers **list comprehensions**: `[x**2 for x in nums]`, `[x for x in nums if x%2==0]`.

---

### `str.join()`
`separator.join(iterable_of_strings)` → glues strings into one, separator between each.

```python
' '.join(['a','b','c'])   # 'a b c'
''.join(['h','i','i'])    # 'hii'
','.join(map(str, [1,2,3]))  # '1,2,3'  (must convert non-strings first)
```
⚠️ Called on the **separator**, not the list. Items **must be strings**.
💡 Mirror image of `split`: `'a,b,c'.split(',')` ↔ `','.join([...])`.

**Used in GPT** — the `decode` function rebuilds text from char list with `''.join(...)`.

---

### `set` — unique, unordered collection
Collection of **unique, unordered** items — no duplicates.

```python
{1, 2, 3}           # set literal
set([1, 1, 2])      # {1, 2}  — dedupes
set("hello")        # {'h','e','l','o'}
set()               # empty set  ⚠️ {} is a DICT, not a set!
```
Key properties:
- **Unordered** → no index/position; `s[0]` → `TypeError` ("not subscriptable"). Displayed order isn't guaranteed.
- **Fast membership** → `x in s` is near-instant *regardless of size*, via **hashing** (jumps directly instead of scanning).
- Set math: `|` union · `&` intersection · `-` difference.

**Set vs List — when to use which:**
| Feature | `list` | `set` |
|---|---|---|
| Keeps order | ✅ | ❌ |
| Allows duplicates | ✅ | ❌ |
| Index access `x[0]` | ✅ | ❌ |
| `in` speed | 🐢 slow (scans) | ⚡ fast (hash) |
| Set math | ❌ | ✅ |

💡 **List** → order/duplicates/positions matter (sequences, tokens, rows). **Set** → uniqueness + fast lookups + set math (dedup, "have I seen this?").

**Used in GPT** — building the vocabulary:
```python
chars = sorted(set(text))     # set(text) → unique chars (dedupe); sorted(...) → ordered, indexable list
vocab_size = len(chars)       # number of distinct token types the model learns
```
⚠️ `set` gets *uniqueness*, `sorted` gives a **stable, indexable** vocabulary — essential for a consistent char ↔ integer mapping (`stoi`/`itos`).

---

## 3. The Data Pipeline (text → tensors)

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Flow:
```
text (string) → encode() → [46,47,47,...] (list[int]) → torch.tensor(...,long) → data (int64 tensor)
```

- `encode(text)` → list of integer token IDs (one per char)
- `torch.tensor(...)` → wraps list into a PyTorch tensor (GPU-capable, gradient-capable)
- `dtype=torch.long` → 64-bit **integers** because these are **token IDs (indices)**, not quantities

`data` = one long 1-D tensor holding the **entire dataset** as integer IDs. Later it's split (train/val), chopped into `block_size` chunks, batched, and fed to the model.

---

## 4. PyTorch Core Concepts

### Why tensors, not Python lists?
1. **Vectorized math** — element-wise ops & matmul in fast C/CUDA, not Python loops.
2. **GPU** — tensor is one contiguous block shippable to thousands of GPU cores.
3. **Gradient tracking (autograd)** — a list has no computational history; a tensor does.

💡 List = filing cabinet of loose numbers. Tensor = spreadsheet the GPU can do calculus on.

### How gradients are tracked (autograd)
1. **Builds a computation graph** as you compute. Each result stores a `grad_fn` pointing back to the op that made it.
2. **`.backward()`** walks the graph in reverse applying the **chain rule**, filling each tensor's `.grad`.
3. **Optimizer** nudges weights: `weight -= lr * weight.grad`.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)   # dy/dx = 2x+3 = 7.0
```
⚠️ Input **data** does NOT require grad (it's fixed). **Weights** do. Gradients flow to weights, never to raw data.

### dtypes: two roles for numbers
| Role | Meaning | dtype |
|---|---|---|
| **Indices / labels** | *which* token/class (pointers into a table) | `int64` (`long`) — always integers |
| **Weights / activations** | the values you do math on | `fp32`, `fp16`, `bf16` |

⚠️ **Interview trap:** why must embedding indices be `long`?
- They're **pointers**, not quantities — "row 46.5" is meaningless, so they must be integers.
- `int64` covers huge vocab/index ranges; `nn.Embedding` & `cross_entropy` are implemented to require `long`.

💡 **fp32 / fp16 / int8 refer to the *weights*, NOT the token indices:**
- **fp32 training** = weights/grads are 32-bit floats (standard, most precise).
- **fp16/bf16 (mixed precision)** = 16-bit floats to save memory/speed.
- **int8** = usually **quantization** — compressing trained fp32 weights to 8-bit for faster/smaller **inference**. True int8 *training* is rare.
- Through all of this, **token indices stay `long`** — different numbers, different job.

```
data (int64 indices) --lookup--> embedding vectors (fp32 floats) --> rest of net (floats)
   "which row?"                         "actual values to do math on"
```

### `torch.manual_seed(n)`
Pins PyTorch's RNG to a fixed starting point → **reproducible** random results (weight init, sampling, dropout).

```python
torch.manual_seed(1337); torch.randn(3)   # same numbers every run
```
💡 Computers use **pseudo-randomness**: a formula producing a random-looking sequence; the **seed = starting point**. Same seed → same sequence.
- `1337` is arbitrary (leet-speak); `42` common. Value doesn't matter, being **fixed** does.
- Only affects PyTorch (Python `random` & NumPy have own seeds).
- Karpathy uses it so your notebook numbers **match his exactly**.

---

## 5. Batching & Sampling

### `torch.randint(low, high, size)`
Random **integers** in range **`[low, high)`** (low included, high excluded).

```python
torch.randint(0, 10, (5,))   # 5 ints from 0..9
```
⚠️ `high` is **exclusive**. If `low` omitted, defaults to 0.

### `torch.stack(list_of_tensors, dim=0)`
Combines same-shape tensors into ONE tensor with a **new** dimension (piling).

```python
a = torch.tensor([1,2,3]); b = torch.tensor([4,5,6])

torch.stack([a,b], dim=0)   # (2,3)  rows:  [[1,2,3],[4,5,6]]
torch.stack([a,b], dim=1)   # (3,2)  cols:  [[1,4],[2,5],[3,6]]
```
`dim` controls **where the new axis is inserted**. `dim=1` result = transpose of `dim=0`.

⚠️ **`stack` vs `cat`:**
| | `stack` | `cat` |
|---|---|---|
| new dimension? | ✅ yes | ❌ no |
| result rank | input+1 | same |
| three `(3,)` tensors | `(3,3)` | `(9,)` |

### `get_batch` — puts it together
```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))   # random start positions
    x = torch.stack([data[i:i+block_size]     for i in ix])     # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])     # targets = x shifted by 1
    return x, y
```
- `randint` → `batch_size` random start indices into the data.
- `len(data) - block_size` prevents running off the end (always a full chunk + 1 for `y`).
- `stack` → turns the list of chunks into one `(batch_size, block_size)` batch tensor.
- 💡 Random sampling = model sees varied, shuffled examples → more stable training & better generalization.
- `y` is `x` shifted one position right (the "next token" targets).

---

## 6. Embeddings

**Embedding = learnable lookup table mapping each integer token ID → a dense vector of floats.**
The bridge between **discrete symbols** and **vectors a neural net can learn from**.

```python
emb = nn.Embedding(vocab_size, n_embd)   # vocab_size rows × n_embd cols (learnable)
emb(torch.tensor([46,47,47]))            # (3, n_embd)  — fetches rows 46,47,47
```
💡 **int-in, float-out**: input = integer IDs ("which rows"), output = float vectors ("their values"). No multiplication at input — it's row addressing.
💡 Vectors start **random**, then gradients nudge them so similar-behaving tokens get similar vectors. **Nobody hand-codes meanings — the model discovers them.**

### Two embedding tables in the full GPT
```python
# 1. Token embedding — WHAT is this token
tok_emb = self.token_embedding_table(idx)              # (B, T, n_embd)
# 2. Positional embedding — WHERE it is in the sequence
pos_emb = self.position_embedding_table(torch.arange(T))  # (T, n_embd)
# combine
x = tok_emb + pos_emb                                  # (B, T, n_embd)
```
⚠️ Attention has **no built-in sense of order** → positional embedding is why
"the dog bit the man" ≠ "the man bit the dog".

Pipeline:
```
text → encode → token IDs (long) → EMBEDDING lookup → float vectors → attention → ... → predictions
```

---

## 7. The Bigram Model

**Predicts the next token from ONLY the single previous token.** "Bi-gram" = 2 tokens.
Ignores all earlier context.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # note: vocab×vocab!

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)   # (B, T, vocab_size)
        ...
```
💡 Table is `(vocab_size, vocab_size)`: feeding token ID `q` → looks up its row → that row **directly is the scores (logits) for every possible next token**. The **lookup IS the prediction** — no hidden layers, no attention, no context.

⚠️ This is NOT meaningful "embedding creation" — rows are next-token score tables (length `vocab_size`). Real meaning-vectors (`n_embd`) come later in the full GPT.

**Why start here?**
1. Simplest complete, trainable, text-generating model (~30 lines).
2. Establishes full **pipeline**: batch → forward → loss → `backward()` → optimizer → generate.
3. Output = plausible char-soup (captures which letters follow which, but no words/grammar).

**Fundamental limitation:** only 1 token of context → can never form real words.
🎯 This weakness ("only sees one token back") is the **entire motivation for attention / Transformers**.

```
Bigram (1 token context) → Self-attention → Multi-head → Transformer blocks → full GPT
                            (each token can look at ALL previous tokens)
```

---

## 8. Embedding Techniques (Landscape)

Two philosophies:
- **Learned inside a bigger model** (GPT) — embedding table = weights trained end-to-end by backprop. No separate step.
- **Standalone dedicated algorithms** — whose only goal is good vectors (most famous named ones).

| Approach | Context-aware? | Examples | Note |
|---|---|---|---|
| **Count / matrix-based** | ❌ No | LSA/LSI, **GloVe** | factorize co-occurrence matrix; `king−man+woman≈queen` |
| **Predictive shallow** | ❌ No | **Word2Vec** (CBOW/Skip-gram), **FastText** | famous "word vectors"; FastText embeds subwords → handles unseen/misspelled words |
| **Contextual (deep)** | ✅ Yes | ELMo, **BERT**, **GPT** | vector depends on whole sentence; fixes "bank" (river vs money) |
| **Sentence-level** | ✅ Yes | SBERT, OpenAI/Cohere/E5/BGE embedding APIs | backbone of search / RAG / vector DBs |
| **Beyond text** | — | Node2Vec/DeepWalk (graphs), CLIP (image+text), rec-sys | idea generalizes everywhere |

🎯 **Interview soundbite:** Most famous *named* embedding techniques = **Word2Vec, GloVe, FastText** (static word vectors). **BERT/GPT** = contextual. GPT has no separate "make embeddings" step — it **learns its `nn.Embedding` table end-to-end**.

**Key evolution:** static one-vector-per-word (Word2Vec/GloVe) → context-dependent vectors learned jointly with the model (BERT/GPT).

---

## 9. Quick Interview Cheat-Sheet

**Q: Why tensors not lists?**
→ Vectorized/GPU math + autograd (computation graph via `grad_fn`, `.backward()`, `.grad`).

**Q: How does PyTorch compute gradients?**
→ Autograd builds a graph during forward pass; `.backward()` applies chain rule in reverse to fill `.grad`; optimizer updates weights.

**Q: Why must token indices be `long` (int64)?**
→ They're pointers into a lookup table, not quantities; can't index with a float. `nn.Embedding`/`cross_entropy` require it; int64 covers large ranges.

**Q: fp32 vs fp16 vs int8 — what do they apply to?**
→ The **weights/activations**, not indices. fp32 = standard training precision; fp16/bf16 = mixed precision (speed/memory); int8 = usually quantization for inference. Indices always stay `long`.

**Q: What is an embedding?**
→ Learnable lookup table: integer token ID → dense float vector. Int-in, float-out. Learned so similar tokens get similar vectors.

**Q: Token vs positional embedding?**
→ Token = *what* the token is; positional = *where* it is (attention is order-blind, so we add position info). Sum feeds the Transformer.

**Q: What is a bigram model & why start with it?**
→ Predicts next token from only the previous one (single `nn.Embedding(vocab,vocab)` where lookup = prediction). Baseline that sets up the training pipeline; its 1-token-context limit motivates attention.

**Q: `stack` vs `cat`?**
→ `stack` adds a new dimension (rank+1); `cat` extends an existing one (same rank).

**Q: `torch.manual_seed`?**
→ Fixes the pseudo-RNG start point for reproducibility (weight init, sampling). Only affects PyTorch's RNG.

**Q: `randint` range?**
→ `[low, high)` — high exclusive; low defaults to 0.

**Q: Why always pass `encoding='utf-8'`?**
→ Files are bytes; encoding maps bytes↔chars. OS default varies → corruption/`UnicodeDecodeError`. UTF-8 = variable-width (1–4 bytes), ASCII-compatible, self-synchronizing.

**Q: UTF-8 vs ASCII?**
→ ASCII = 128 chars, 1 byte, English only. UTF-8 = superset covering all Unicode; identical bytes for plain English.

**Q: `set` vs `list`?**
→ set = unique + unordered + O(1) `in` (hashing) + set math, no indexing. list = ordered + duplicates + indexable. `sorted(set(text))` = dedupe then make an ordered, indexable vocab.

**Q: `set()` vs `{}`?**
→ `set()` is an empty set; `{}` is an empty **dict**.

---

*This course section will grow as it continues: self-attention, multi-head attention, Transformer blocks, layer norm, residual connections, training loop, generation.*

---

# 🧩 Standalone Topics

> Deep-dives studied outside a specific course. Backlog items graduate here once written up.

_(none yet)_
