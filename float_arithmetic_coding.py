from mpmath import mp 
import mpmath, math
from utils import load_text
from distribution import Distribution, DistFromModel

# Uniform distribution
def uniform(item: str, string: list=[]) -> float: 
    return 1/len(alphabet)

def encode(sequence: str, dist: Distribution) -> float:
    sequence = list(sequence) + ["<END>"]

    # Entropy of data stream: H(X) = -Σp(x)log2[p(x)]
    # Given a uniform distribution, p(x) = 1/#X ∀ x and Σp(x) = 1
    # Thus, H(X) = (-Σlog2[p(x)])/#X = -log2[p(x)]
    
    #mp.prec = 100000
    
    start = mp.mpf(0)
    end = mp.mpf(1)
    
    alphabet = dist.get_alphabet()
    mp.prec = -math.ceil(len(sequence)*math.log2(1/len(alphabet)))
    print(f"Precision: {mp.prec} bits")
    for c in range(len(sequence)): 
        location = alphabet.index(sequence[c])
        new_start = start
        dist_given = dist.p_given(sequence[:c])
        for i in range(location):
            new_start += mp.mpf(dist_given[alphabet[location]]) * (end-start)
        
        end = new_start + mp.mpf(dist_given[alphabet[location]]) * (end-start)
        start = new_start

        req_bits = -mpmath.ceil(mpmath.log(end-start, 2)) + 10
        if mp.prec < req_bits: 
            mp.prec = req_bits
        
    return (start + end) / mp.mpf(2)


def decode(code: float, dist: Distribution) -> str:

    start = 0
    end = 1
    result = []
    alphabet = dist.get_alphabet()
    while end - start > mpmath.power(2, mp.mpf(-mp.prec)) and (len(result)==0 or result[-1] != "<END>"): 
        new_start = start
        for i in range(len(alphabet)): 
            symbol_range = mp.mpf(dist.p(alphabet[i], result)) * (end-start)
            if new_start + symbol_range > code: 
                end = new_start + symbol_range
                result.append(alphabet[i])
                #print(alphabet[i], end='')
                break
            new_start += symbol_range
        start = new_start
        

    return "".join(result[:-1])

distribution = DistFromModel("modelling/hemingway.pth")

#print(decode(mp.mpf("0.739552692427511692180069615447330809960907787386281026127356076143241967236"), distribution))
#exit(0)

string = load_text("sentence").lower()
print(f"Original: \"\"\"\n{string}\n\"\"\"")
#alphabet = list("\n\40!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~") + ["<END>"]

encoded = encode(string, distribution)
print(f"Compress: {encoded}")
decoded = decode(encoded, distribution)
print(f"Uncompress: \"\"\"\n{decoded}\n\"\"\"\n{"Yay! Lossless compression!" if string == decoded else "Oh no! Lossy compression!"}")

