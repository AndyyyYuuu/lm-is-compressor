# Language Model-Based Text Compressor

This repository is a demonstration of using a language model to perform lossless compression on natural langage. Whether you're a novice programmer or experienced computer scientists, I hope you'll find this experiment interesting/informative. 

> ## Table of Contents
> 1. [Run the Experiment](#run-the-experiment)
> 2. [Experimental Results](#experimental-results)
> 3. [Explanation](#explanation)
> 4. [Methods](#methods)
> 5. [Credits](#credits)
> 6. [Further Reading](#further-reading)
---
## Run the Experiment
### Prerequisites
- Python 3.11.1
- Required packages: `torch`, `transformers`

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
## Experimental Results
I used [Weights & Biases](wandb.org) to log and create graphs for my results. 

I tested the compression rates of 4 text files of varying size. See [`testing.py`](testing.py) for testing code and [`texts`](texts) for the text files used. 

<img width="700" alt="negative_log_2_prob" src="https://github.com/user-attachments/assets/003f919c-035e-40c1-9218-d2d3c0915a1c" />

From the graph, we see that compression rate is around 20% for large text files. This means that the compressor is able to reduce english text files to one-fifth of their original size. 


---
## Explanation
### Theoretical Feasibility

Before explaining data compression, we must first explain our measure of information. i.e. How much of a text file is redundant, compressible data, and how much of it is essential? 

In [The Mathematical Theory of Communication](https://pure.mpg.de/rest/items/item_2383164_3/component/file_2383163/content), Claude E. Shannon proposes the following measure of information content of a probabilistic event:  

$$I(x) = -log_b{p(x)}$$

$$b$$ is the base, dictating the unit used for our output. $$b=2$$ yields an output of information in bits. 

Here is a graph of I(x) when $$b=2$$. 

<img width="300" alt="negative_log_2_prob" src="https://github.com/user-attachments/assets/d736f201-2ef2-4bf8-a46e-ff96456a521c" />

As the probability $$p(x)$$ of an event $$x$$ decreases, the information content or level of "surprise" of an event increases. 

Hence **Shannon entropy**, or expected information content, of a distribution is given when you take the expected value of all the information contents of the events possible. 

$$H(P) = -\sum_x{P(x)log_2{P(x)}}$$

Shannon entropy defines the theoretical lower bound of the size of your compressed data. Unfortunately, when it comes to compressing English text, we cannot know the true distribution $$p$$ of every possible sentence. Instead, we might use a distribution $$\hat{P}$$ as a way to estimate P. We can update the length of our compressed file to the following: 

$$H(P, \hat{P}) = -\sum_x{P(x)log_2{\hat{P}(x)}}$$

Our estimated information content for each event is now $$log_2{\hat{P}(x)}$$. Yet we still use $$P(x)$$ to weight events by probability because, in the real world, the true probabilities of each event is still going to be $$P(x)$$. This measure is known as **cross-entropy**, and it represents the length of our compressed file when using an estimated distribution $$\hat{P}$$. 

If we were to compress a file efficiently, we need to minimize the length of our output file and hence the value of cross-entropy $$H(P, \hat{P})$$. Some of you may know that cross-entropy approaches a minimum when the two distributions approach equality. You can also derive this from a bit of calculus, but since nobody wants to read that, you're just gonna have to take my word for it.

Making $$\hat{P}(x)$$, the estimated distribution of texts in the English language, as close as possible to $$P(x)$$, the actual distribution of English, might sound familiar. That's because it is the exact functionality of a large language model. In other words, the more accurate a large language model is, the lower its cross-entropy, leading to tighter compression rates. 

---
## Methods
### Arithmetic Coding
### Language Model
This project uses GPT-2 from `huggingface` to run its compressor. 
### Chunk Processing

### KV-Caching
KV-caching was used to accelerate language model inference. Given a chunk size of 2048 characters (amounting to around 400-500 tokens for typical text), the addition of KV-caching can decrease inference time by around 80%. 


---
## Credits

- Special thanks to [Qihang Zhang](https://github.com/Qihang-Zhang) for his mentorship and guidance throughout this project. 
- This repository uses [nayuki/Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding) as a submodule to perform arithmetic coding.
---
## Further Reading
#### Papers on this topic
- **[Language Modeling is Compression](https://arxiv.org/abs/2309.10668) by Delétang et al.**  
  - Experiments with the use of language models in text, image, and audio compression through similar methods as the ones used in this repository. Also shows that a compressor can be used to build a conditional generative model. 

#### Information theory
- [Elements of Information Theory](https://cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf)
  - Chapter 1 describes the foundational concept of Shannon entropy and its variations.
  - Chapter 5 outlines the use of Information Theory in data compression. 
