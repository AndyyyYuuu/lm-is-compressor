
# Uniform distribution
def p_given(item: str, string: list) -> float: 
    return 1/len(alphabet)

def encode(sequence: str) -> float:
    sequence = list(sequence) + ["END"]
    #alphabet = sorted(list(set(sequence))) + ["END"]
    start = 0
    end = 1
    for c in range(len(sequence)): 
        location = alphabet.index(sequence[c])
        new_start = start
        for i in range(location):
            new_start += p_given(alphabet[i], sequence[:c]) * (end-start)
        
        end = new_start + p_given(alphabet[location], sequence[:c]) * (end-start)
        start = new_start
    return (start + end) / 2


def decode(code: float) -> str:

    start = 0
    end = 1
    result = []
    while end - start > 0 and (len(result)==0 or result[-1] != "END"): 
        new_start = start
        for i in range(len(alphabet)): 
            symbol_range = p_given(alphabet[i], result) * (end-start)
            if new_start + symbol_range > code: 
                end = new_start + symbol_range
                result.append(alphabet[i])
                break
            new_start += symbol_range
        start = new_start
    return "".join(result[:-1])

string = "claude shannon"

alphabet = sorted(list(set(string))) + ["END"]
encoded = encode(string)
decoded = decode(encoded)
print(f"Original: {string} \nCompress: {encoded} \nUncompress: {decoded}\n{"Yay! Lossless compression!" if string == decoded else "Oh no! Lossy compression!"}")
