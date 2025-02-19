import transformers, torch, contextlib, importlib, math
from tqdm import tqdm

arithmeticcoding = importlib.import_module("Reference-arithmetic-coding.python.arithmeticcoding")

gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
ALPHABET_SIZE = 50257
CONTEXT_LENGTH = 1024
SCALING_FACTOR = 10000

def p_given(condition, model):
	input_ids = torch.tensor(condition).unsqueeze(0)
	with torch.no_grad():
		outputs = model(input_ids, labels=input_ids)
	token_log_prob = torch.log_softmax(outputs.logits[0, -1], dim=-1)
	prob = torch.exp(token_log_prob)
	#scaling_factor = 1/torch.min(prob)
	freqs = prob * SCALING_FACTOR
	return [math.ceil(float(i)) for i in freqs] + [1]


def compress(inp, bitout):
	bos = torch.tensor([gpt2_tokenizer.bos_token_id])
	eos = torch.tensor([gpt2_tokenizer.eos_token_id])
	print(bos, eos)
	inp = torch.cat((bos, gpt2_tokenizer.encode(inp.decode(), return_tensors="pt")[0], eos))
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

	# TODO: Infer from entire sequence for batch processing? 
	for i in tqdm(range(1, len(inp)), unit=" tokens"):
		
		context = inp[:i]
		if len(context) > CONTEXT_LENGTH: 
			context = context[len(context)-CONTEXT_LENGTH:]
		new_freqs = p_given(context, gpt2_model)
		for j in range(ALPHABET_SIZE): 
			freqs.set(j, new_freqs[j])

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
		new_freqs = p_given(context, gpt2_model)
		for i in range(ALPHABET_SIZE): 
			freqs.set(i, new_freqs[i])
		# Decode and write one token
		symbol = dec.read(freqs)

		if symbol == gpt2_tokenizer.eos_token_id:
			break

		context.append(symbol)
		if len(context) > CONTEXT_LENGTH: 
			context.pop(0)
		
		# Iterate through characters, then bits
		for c in gpt2_tokenizer.decode(symbol): 
			for i in range(8):
				out.write((ord(c) >> (7 - i)) & 1)
		
		tokens_count += 1
		print(f"Decompressed {tokens_count} tokens\r", end="")
	print(f"Finished decompressing {tokens_count} tokens")


def compress_file(input_path, output_path): 
	with open(input_path, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		compress(inp.read(), bitout)

def decompress_file(input_path, output_path): 
	with open(input_path, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		decompress(inp, bitout)

compress_file("texts/sentence.txt", "tests/lm_output.bin")
decompress_file("tests/lm_output.bin", "tests/restore.txt")