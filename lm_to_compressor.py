import transformers, torch, contextlib, importlib, math
from tqdm import tqdm

arithmeticcoding = importlib.import_module("Reference-arithmetic-coding.python.arithmeticcoding")

torch.set_default_dtype(torch.float64)

gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.eval()

ALPHABET_SIZE = 50257
CONTEXT_LENGTH = 1024
SCALING_FACTOR = 10000

def p_given(condition, model):
	input_ids = torch.tensor(condition).unsqueeze(0)
	with torch.no_grad():
		outputs = model(input_ids, labels=input_ids)
	
	token_log_prob = torch.log_softmax(outputs.logits[0, -1], dim=-1)
	
	prob = torch.exp(token_log_prob)
	freqs = prob * SCALING_FACTOR
	return torch.ceil(freqs).to(torch.int)

def batch_p_given(condition, model):
	# Condition is a tensor of shape (sequence length, )
	
	input_ids = torch.tensor(condition).unsqueeze(0)

	with torch.no_grad():
		outputs = model(input_ids)
	
	token_log_prob = torch.log_softmax(outputs.logits.squeeze(0), dim=-1)

	prob = torch.exp(token_log_prob)
	freqs = prob * SCALING_FACTOR
	return torch.ceil(freqs).to(torch.int)


def compress(inp, bitout):
	# TODO: Process in chunks
	bos = torch.tensor([gpt2_tokenizer.bos_token_id])
	eos = torch.tensor([gpt2_tokenizer.eos_token_id])
	print(bos, eos)
	inp = torch.cat((bos, gpt2_tokenizer.encode(inp.decode(), return_tensors="pt")[0], eos))
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
	print("Computing probabilities ...", end="", flush=True)
	dists = batch_p_given(inp, gpt2_model)
	print("\r\033[KProbabilities computed!", flush=True)

	for i in tqdm(range(1, len(inp)), unit=" tokens"):
		new_freqs = dists[i-1]
		for j in range(ALPHABET_SIZE): 
			freqs.set(j, int(new_freqs[j].item()))

		enc.write(freqs, inp[i].item())

	enc.finish()

def decompress(inp, out):
	bitin = arithmeticcoding.BitInputStream(inp)
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
	context = [gpt2_tokenizer.bos_token_id]
	tokens_count = 0

	while True: 
		#new_freqs = p_given(context, gpt2_model)
		# TODO: KV cache
		new_freqs = batch_p_given(torch.tensor(context), gpt2_model)[-1]
		for i in range(ALPHABET_SIZE): 
			freqs.set(i, int(new_freqs[i]))
		# Decode and write one token
		symbol = dec.read(freqs)

		# TODO: Use more robust stopping condition
		# Get cross entropy of the entire sequence by summing the cross entropy of each token
		# Cross entropy of each token is -log2(prob of the token)
		# When ceil(Cross entropy) > code length, stop
		if symbol == gpt2_tokenizer.eos_token_id:
			break

		context.append(symbol)
		if len(context) > CONTEXT_LENGTH: 
			context.pop(0)
		
		# Iterate through characters, then bits
		for c in gpt2_tokenizer.decode(symbol): 
			for i in range(8):
				out.write((ord(c) >> (7 - i)) & 1)
		out.output.flush()
		tokens_count += 1
		print(f"\rDecompressed {tokens_count} tokens", end="")
	print(f"\rFinished decompressing {tokens_count} tokens")


def compress_file(input_path, output_path): 
	with open(input_path, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		compress(inp.read(), bitout)

def decompress_file(input_path, output_path): 
	with open(input_path, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		decompress(inp, bitout)

compress_file("texts/paper.txt", "tests/lm_output_2.bin")
decompress_file("tests/lm_output_2.bin", "tests/restore.txt")