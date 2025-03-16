import transformers, torch, contextlib, importlib, math, os, re, shutil, time, argparse
from tqdm import tqdm

arithmeticcoding = importlib.import_module("Reference-arithmetic-coding.python.arithmeticcoding")

torch.set_default_dtype(torch.float64)

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

ALPHABET_SIZE = 50257
SCALING_FACTOR = 10000
CHUNK_SIZE = 2048

def p_given(input_id, model, kv_cache=None):
	input_id = input_id.unsqueeze(0)
	with torch.no_grad():
		outputs = model(input_id, past_key_values=kv_cache, use_cache=True)
		kv_cache = outputs.past_key_values
	
	token_log_prob = torch.log_softmax(outputs.logits[0, -1], dim=-1)
	
	prob = torch.exp(token_log_prob)
	freqs = prob * SCALING_FACTOR
	return torch.ceil(freqs).to(torch.int), kv_cache

def batch_p_given(condition, model):
	# Condition is a tensor of shape (sequence length, )
	
	input_ids = condition.unsqueeze(0)

	with torch.no_grad():
		outputs = model(input_ids)
	
	prob = torch.softmax(outputs.logits.squeeze(0), dim=-1)
	freqs = prob * SCALING_FACTOR
	return torch.ceil(freqs).to(torch.int)

def flat_p(condition):
	return torch.ones(len(condition), ALPHABET_SIZE)


def compress(inp: str, bitout):
	# TODO: Process in chunks
	bos = torch.tensor([tokenizer.bos_token_id])
	inp = torch.cat((bos, tokenizer.encode(inp, return_tensors="pt")[0]))
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
	#dists = batch_p_given(inp, model)
	dists = flat_p(inp)

	for i in tqdm(range(1, len(inp)), unit=" tokens", leave=False):
		new_freqs = dists[i-1]
		for j in range(ALPHABET_SIZE): 
			freqs.set(j, int(new_freqs[j].item()))

		enc.write(freqs, inp[i].item())

	enc.finish()

def decompress(inp, inp_length, out):
	bitin = arithmeticcoding.BitInputStream(inp)
	initfreqs = arithmeticcoding.FlatFrequencyTable(ALPHABET_SIZE)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
	last_token = tokenizer.bos_token_id
	tokens_count = 0
	kv_cache = None
	info_content = 0
	with tqdm(total=inp_length, unit=" bits", leave=False) as progress:
		while True: 
			new_freqs, kv_cache = p_given(torch.tensor([last_token]), model, kv_cache)

			for i in range(ALPHABET_SIZE): 
				freqs.set(i, int(new_freqs[i]))
			# Decode and write one token
			symbol = dec.read(freqs)

			# Get info content of the entire sequence by summing the info content of each token
			# Info content of each token is -log2(prob of the token)
			# When ceil(info content) > code length, stop
			last_info_content = info_content
			info_content += -math.log2(new_freqs[symbol]/new_freqs.sum())
			if math.ceil(info_content) > inp_length:
				break
			
			progress.update(math.ceil(info_content) - math.ceil(last_info_content))
			
			last_token = symbol
			
			# Iterate through characters, then bits
			for c in tokenizer.decode(symbol): 
				for i in range(8):
					out.write((ord(c) >> (7 - i)) & 1)
			out.output.flush()
			tokens_count += 1


def compress_file(input_path, output_path): 
	with open(input_path, "rb") as inp:
		txt = inp.read().decode("utf-8")
		chunks = [txt[i:i+CHUNK_SIZE] for i in range(0, len(txt), CHUNK_SIZE)]
		if os.path.exists(output_path):
			shutil.rmtree(output_path)
		os.makedirs(output_path)
		for i, chunk in enumerate(tqdm(chunks, unit=" chunks")):
			with contextlib.closing(arithmeticcoding.BitOutputStream(open(os.path.join(output_path, f"{i}.bin"), "wb"))) as bitout:
				compress(chunk, bitout)


def decompress_file(input_path, output_path): 
	with contextlib.closing(arithmeticcoding.BitOutputStream(open(output_path, "wb"))) as bitout:
		files = [f for f in os.listdir(input_path) if f.endswith(".bin")]
		files.sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
		for file in tqdm(files, total=len(files), unit=" chunks"):
			with open(os.path.join(input_path, file), "rb") as inp:
				inp_length = os.path.getsize(os.path.join(input_path, file)) * 8
				decompress(inp, inp_length, bitout)

def main():
	parser = argparse.ArgumentParser(description="Compress or decompress files.")
	parser.add_argument("-c", "--compress", action="store_true", help="Compress a file.")
	parser.add_argument("-d", "--decompress", action="store_true", help="Decompress a file.")
	parser.add_argument("input", help="The input file to compress or decompress.")
	parser.add_argument("output", help="The output file to save compressed or decompressed data.")
	args = parser.parse_args()

	if args.compress:
		compress_file(args.input, args.output)
	elif args.decompress:
		decompress_file(args.input, args.output)

if __name__ == "__main__":
	main()
