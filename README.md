# Language Model-Based Text Compressor

This repository is a demonstration of using a language model to perform lossless compression on natural langage. It shows that the use of GPT-2 as part of a text compression algorithm, albeit extremely slow to run, can achieve a compression rate of around 20% on English natural language. Whether you're a novice programmer or experienced computer scientists, I hope you'll find this experiment interesting/informative. 

> ## Table of Contents
> 1. [Run the Experiment](#run-the-experiment)
> 2. [Theoretical Foundation](#theoretical-foundation)
> 3. [Implementation](#implementation)
> 4. [Experimental Results](#experimental-results)
> 5. [Credits](#credits)
> 6. [Further Reading](#further-reading)
---
## Run the Experiment
### Prerequisites
- Python 3.11.1
- Required packages: `torch`, `transformers`, `tqdm`
- Optional package: `wandb`

### Setup
1: Clone repository
```bash
git clone https://github.com/AndyyyYuuu/lm-is-compressor.git
cd lm-is-compressor
```
2: Install dependencies
```bash
pip install -r requirements.txt
```
Optionally, use a [virtual environment](https://gist.github.com/ryumada/c22133988fd1c22a66e4ed1b23eca233).
### Usage
Run a compression experiment: 
```bash
python3 lm_to_compressor.py --compress {input text path} {output code path}
```

Run a decompression experiment: 
```bash
python3 lm_to_compressor.py --decompress {code path} {output text path}
```

---
## Theoretical Foundation

Before explaining data compression, we must first explain our measure of information. i.e. How much of a text file is redundant, compressible data, and how much of it is essential. 

**Shannon entropy**, or expected information content, of a distribution is given by taking the expected value of all the information contents of possible events. 

$$H(P) = -\sum_x{P(x)log_2{P(x)}}$$

Shannon entropy defines the theoretical lower bound of the size of the compressed file, per unit of the uncompressed file (oftentimes bytes, but LLM tokens in this project). In otherwords, $H(P) \times input \space file \space length \leq expected \space output \space file \space size$. 

Unfortunately, when it comes to compressing English text, we cannot know the true distribution $p$ of every possible sentence. Instead, we might use a distribution $P_\theta$ as a way to estimate $P$. We can update the lower bound for the length of our compressed file to the following, known as cross-entropy: 

$$H(P, P_\theta) = -\sum_x{P(x)log_2{P_\theta(x)}}$$

If we were to compress a file efficiently, we need to minimize the length of our output file and hence the value of cross-entropy $H(P, P_\theta)$. 

It also happens that cross-entropy is precisely the metric that is minimized in the pre-training process of modern Transformer language models. In that scenario, $P_\theta$ becomes our language model, which, as a result of optimization, is already in a state to minimize cross-entropy and thus the lower bound of the output code length. In other words, the more accurate a large language model is, the more effectively it compresses text. 

---
## Implementation
The following section details the methods used to achieve the aforementioned task of converting a language model into a data compressor. 

### Arithmetic Coding

Arithmetic coding works by encoding every possible combination of symbols into a single number between 0 and 1. 


The code length of arithmetic coding is exactly 

$$\lceil -\sum_i^{\#d}{log_2{P_i(x)}} \rceil$$

where $P_i(x)$ is the conditional probability of token $i$ given by the language model. 

### Language Model

This project uses [GPT-2 from `huggingface`](https://huggingface.co/openai-community/gpt2) to run its compressor. 



In the source code, the distributions of possible next tokens are computed for every token simultaneously by inputting the entire string into the LLM. 

### Chunk Processing
GPT-2 has a limited context length and will throw an error if you attempt to calculate distributions for sequences longer than 1024 tokens. Therefore, I split longer text files into chunks of 1023 tokens, one less than 1024 to make space for an extra `<|endoftext|>` token before each chunk. Each compressed chunk is placed in a `.bin` file inside a folder representing a compressed file. 

While the splitting of input text definitely decreases the capabilities of the language model by limiting its context, 

An alternative solution would have been to implement a 1024-token moving window, ensuring the LLM always has 1024 tokens of context to work with. Unfortunately, the aforementioned batch inference would not have been possible with this solution. 

### KV-Caching
KV-caching was used to accelerate language model inference. 

In a transformer, the three numbers key (K), query (Q), and value (V) are computed for each input token. K and V are 

Given a chunk size of 2048 characters, the addition of KV-caching can decrease inference time by around 80%. 

<img width="700" alt="kv_cache_effects" src="https://github.com/user-attachments/assets/c13198ae-22db-4f3d-8048-a8e56f8cf720" />

As seen from the graph, the algorithm without KV-caching experiences a linear increase in decoding time per token. The number of tokens for the language model to process increases for every additional token until the decoder moves on to the next chunk of data. On the other hand, KV-caching allows each inference to be performed in a relatively fixed amount of time. 

---
## Experimental Results
These experiments on my data compressor were run using the files in [`texts`](texts). I used [Weights & Biases](wandb.org) and Google Sheets to log and create graphs for my results. 
### Compression Rate for Natural Language
I tested the compression rates of 4 text files of varying size. See [`testing.py`](testing.py) for testing code. 

<img width="500" alt="Compression Rates on Various Natural Language Texts" src="https://github.com/user-attachments/assets/54a26c37-3243-4a08-8abd-a19d9800f5aa" />

The x-axis of the graph represents the compression rate of the compressor on various files (compression rate is compressed file size / uncompressed file size). 
From the graph, we see that compression rate is around 20% for large text files. This means that the compressor is able to reduce english text files to one-fifth of their original size. The large compression rate of `sentence.txt` is likely due to decreased language modeling accuracy as a result of the small amount of context. 

### Comparison With Other Methods & Models

![Compression Rates of Various Methods for Email, Article, and Paper](https://github.com/user-attachments/assets/a82f7044-788e-4ab5-9b63-c028c9e064fc)

For the natural language files in [`texts`](texts), GPT-2 arithmetic coding consistently outperforms ZIP in compression rate. It also outperforms the use of weaker models for arithmetic coding, providing evidence for our theoretical proof that a better language model does indeed correspond to a better text compressor. 

I notice that the compression rate of ZIP seems to decrease as file size increases. I will run further tests on larger files. 

---
## Credits

- Massive thanks to [Qihang Zhang](https://github.com/Qihang-Zhang) for his mentorship and guidance throughout this project. 
- This repository uses [nayuki/Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding) to perform arithmetic coding.
- This repository uses [GPT-2](https://huggingface.co/openai-community/gpt2) from Hugging Face. The model was originally introduced in "[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)" by Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 

---
## Further Reading
- **[Language Modeling is Compression](https://arxiv.org/abs/2309.10668) by Delétang et al.**  
  - Experiments with the use of language models in text, image, and audio compression through similar methods as the ones used in this repository. Also shows that a compressor can be used to build a conditional generative model. 
- **[The Mathematical Theory of Communication](https://pure.mpg.de/rest/items/item_2383164_3/component/file_2383163/content) by Claude E. Shannon and Warren Weaver**
  - Recommend reading Warren Weaver's overview of Shannon's theory. 
- **[Elements of Information Theory](https://cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf) by Joy A. Thomas and Thomas M. Cover**
  - Chapter 1 describes the foundational concept of Shannon entropy and its variations. 
  - Chapter 5 outlines the information theory of data compression, including a derivation of Shannon entropy. 
