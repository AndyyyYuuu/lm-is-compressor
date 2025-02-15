import transformers, torch, contextlib, importlib

arithmeticcoding = importlib.import_module("Reference-arithmetic-coding.python.arithmeticcoding")

gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
ALPHABET_SIZE = 50257

def p_given(condition, model):
	input_ids = torch.tensor(condition).unsqueeze(0)
	with torch.no_grad():
		outputs = model(input_ids, labels=input_ids)
	token_log_prob = torch.log_softmax(outputs.logits[0, -1], dim=-1)
	prob = torch.exp(token_log_prob)
	scaling_factor = 1/torch.min(prob)
	freqs = prob * scaling_factor
	return [round(float(i)) for i in freqs] + [1]


def compress(inp, bitout):
	inp = gpt2_tokenizer.encode(inp.decode(), return_tensors="pt")[0]
	
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE+1)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(64, bitout)
	for i in range(len(inp)):
		enc.write(freqs, inp[i].item())
		new_freqs = p_given(inp[:i+1], gpt2_model)
		for i in range(ALPHABET_SIZE+1): 
			freqs.set(i, new_freqs[i])

	enc.write(freqs, ALPHABET_SIZE)  # EOF
	enc.finish()


def decompress(inp, out):
	bitin = arithmeticcoding.BitInputStream(inp)
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE+1)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	dec = arithmeticcoding.ArithmeticDecoder(64, bitin)
	output = []
	while True: 
		# Decode and write one token
		symbol = dec.read(freqs)
		
		if symbol == ALPHABET_SIZE:  # EOF
			break

		output.append(symbol)
		new_freqs = p_given(output, gpt2_model)
		for i in range(ALPHABET_SIZE+1): 
			freqs.set(i, new_freqs[i])

		# Iterate through characters, then bits
		for c in gpt2_tokenizer.decode(symbol): 
			#print(bytes((ord(c),)))
			for i in range(8):
				out.write((ord(c) >> (7 - i)) & 1)


def compress_file(input_path, output_path): 
	with open(input_path, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		compress(inp.read(), bitout)

def decompress_file(input_path, output_path): 
	with open(input_path, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		decompress(inp, bitout)

compress_file("texts/sentence.txt", "output.bin")
decompress_file("output.bin", "restore.txt")