
<div align="center">
  <h1>Language Model-Based Text Compression</h1>
    <i>An accurate language model can be converted into a high-compression, lossless text compressor, and vice versa. </i><br><br>
</div>

This repository is a demonstration of using a language model to perform lossless compression on natural langage. In it, I show that the use of GPT-2 Small as part of an arithmetic coding algorithm, albeit extremely slow to run, can achieve a compression rate of **around 20%** on English natural language. I also demonstrate that this compression rate can be improved through the use of more accurate language models. I hope you'll find this experiment interesting or informative (pun initially not intended). 

> ## Table of Contents
> 1. [Run the Experiment](#run-the-experiment)
> 2. [Theoretical Foundation](#theoretical-foundation)
> 3. [Arithmetic Coding](#arithmetic-coding)
> 4. [Implementation](#implementation)
> 5. [Experimental Results](#experimental-results)
> 6. [Limitations & Further Experiments](#limitations--further-experiments)
> 7. [Credits](#credits)
> 8. [Further Reading](#further-reading)
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
Compress a file: 
```bash
python3 lm_to_compressor.py --compress {input text path} {output code path}
```

Decompress a file: 
```bash
python3 lm_to_compressor.py --decompress {code path} {output text path}
```

To calculate compression rate with optional [Weights & Biases](https://wandb.ai/) logging, modify settings in `testing.py`, then run: 
```bash
python3 testing.py
```


---
## Theoretical Foundation

Before explaining data compression, we must first explain our measure of information. i.e. How much of a text file is redundant, compressible data, and how much of it is essential. 

**Shannon entropy**, or expected information content, of a distribution is given by taking the expected value of all the information contents of possible events. 

$$H(P) = -\sum_x{P(x)log_2{P(x)}}$$

Shannon entropy defines the theoretical lower bound of the size of the compressed file, per unit of the uncompressed file (oftentimes bytes/characters, but LLM tokens in this project). In otherwords, $H(P) \times input \space file \space length \leq expected \space output \space file \space size$. 

Unfortunately, when it comes to compressing English text, we cannot know the true distribution $p$ of every possible sentence. Instead, we might use a distribution $P_\theta$ as a way to estimate $P$. We can update the lower bound for the length of our compressed file to the following, known as **cross-entropy**: 

$$H(P, P_\theta) = -\sum_x{P(x)log_2{P_\theta(x)}}$$

If we were to compress a file efficiently, we need to minimize the length of our output file and hence the value of cross-entropy $H(P, P_\theta)$. 

It also happens that cross-entropy is precisely the metric that is minimized in the pre-training process of modern Transformer language models. In that scenario, $P_\theta$ becomes our language model, which, as a result of optimization, is already in a state to minimize cross-entropy and thus the lower bound of the output code length. In other words, the more accurate a large language model is, the more effectively it compresses text. 

---

## Arithmetic Coding

Arithmetic coding is employed as the base algorithm to perform text compression. 

Arithmetic coding works by mapping every possible sequence of symbols onto the number line between 0 and 1. We can thus use a single float to represent any sequence as a whole. Arithmetic coding divides the number line into a segment for each possible value of symbol 1. These divisions are further subdivided to encode symbol 2, and so on. 

<img width="855" alt="Arithmetic coding space with uniform distribution" src="https://github.com/user-attachments/assets/7fb9280c-1f78-4705-9430-b94ed2fc3a79" />

This diagram shows an arithmetic coding algorithm with three symbols: "Hello", "world", and "!". The black bars at the bottom represent the intervals on the number line that encode each of the specified messages. 

It is important to note that any float within this interval is able to represent its message. The larger the interval, the more tolerence we have for the precision of this float. Thus, larger intervals correspond to less characters used to encode the message, while smaller intervals correspond to more characters. 

One of the principles of data compression is that we want more frequently occuring symbols to use less characters after compression. We can thus assign larger intervals to more likely sequences by **scaling our interval sizes with their corresponding probabilities**. These probabilities can be computed with a language model. After the addition of a language model, our algorithm might look more like this: 

<img width="855" alt="Arithmetic coding space with conditional distributions" src="https://github.com/user-attachments/assets/97b89225-4bf5-479e-b9c9-eadeba85595d" />

As seen, sequences such as "Hello world !" are assigned much larger intervals than "! world !" and therefore shorter code length. Since "Hello world !" is much more likely, assigning it a shorter code length allows us to decrease the average length of our encoded files in the long run. 

A simplified algorithm for arithmetic coding with conditional probabilities of each symbol is shown below. The loop encodes a list of items sequentially, narrowing down the interval until it is small enough to uniquely represent the string. 
```py
def p_given(symbol: str, context: list) -> float:
    """Not implemented. Returns P(symbol | context)."""

def encode(sequence: list, alphabet: list) -> float:
  start = 0
  end = 1
  
  for c in range(len(sequence)): 
      location = alphabet.index(sequence[c])
      new_start = start

      for i in range(location):
          new_start += p_given(alphabet[i], sequence[:c]) * (end-start)  # scale down conditional probabilities based on current interval
      
      end = new_start + p_given(alphabet[location], sequence[:c]) * (end-start)
      start = new_start
  return (start + end) / 2  # take the midpoint of the interval
```
<!--
The code length of arithmetic coding is exactly 

$$\lceil -\sum_i{log_2{P_i(x)}} \rceil$$

where $P_i(x)$ is the conditional probability of token $i$ given by the language model. 
-->

---

## Implementation
The following section details the methods used to achieve the aforementioned task of converting a language model into a data compressor. 

### Arithmetic Coding

This project uses [nayuki/Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding) to perform arithmetic coding. The above algorithm is heavily constrained by the precision of Python floats. [nayuki/Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding) overcomes this by using Python integer types as representations of floats. Wondrously, Python `int`s have functionally no upper limit. 

### Language Model

This project primarily uses [GPT-2 from `huggingface`](https://huggingface.co/openai-community/gpt2) to run its compressor. 

In the source code, the distributions of possible next tokens are computed for every token simultaneously by inputting the entire string into the LLM. An `<|endoftext|>` token is placed before the string to give the language model context upon which it builds the first character's distribution. 

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

For the natural language files in [`texts`](texts), GPT-2 Small arithmetic coding consistently outperforms ZIP in compression rate. It also far outperforms weaker language models, such as a static uniform distribution, on arithmetic coding. 

I notice that the compression rate of ZIP seems to decrease as file size increases. I will run further tests on larger files. 

![Compression Rates of GPT-2 Sizes](https://github.com/user-attachments/assets/1ca87b8e-2439-40bd-a38a-0c431ec6e267)

I also compared the compression rates of GPT-2 [Small](https://huggingface.co/openai-community/gpt2), [Medium](https://huggingface.co/openai-community/gpt2-medium), and [Large](https://huggingface.co/openai-community/gpt2-large) models, finding that for the three sample texts, larger, more capable versions of the model always outperform their smaller counterparts. This is consistent with the [theory](#theoretical-foundation) hypothesizing a positive correlation between LM accuracy and file reduction. 

---
## Limitations & Further Experiments
This data compressor is extremely slow. Arithmetic coding is a major efficiency bottleneck for each run. A possible solution is to integrate my Python code with arithmetic coding written in C. 

Because of this issue, the 3 texts I used for this experiment are very small on the scale of data compression, as larger texts would simply take too long. The models I used also needed to be small enough to run locally on my MacBook without crashing it. 

I invite any readers with sufficient time / hardware to try running this with larger models or datasets. 

---
## Credits
- Developed and written by [Andy S. Yu](https://github.com/AndyyyYuuu). 
- Massive thanks to [Qihang Zhang](https://github.com/Qihang-Zhang) for his invaluable mentorship and guidance throughout this project. 
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
