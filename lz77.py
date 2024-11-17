
def compress(string: str, window_size: int) -> list: 
    string = list(string)
    result = []
    search_buffer = []
    lookahead_buffer = string[:window_size]
    string = string[window_size:]

    while len(lookahead_buffer) > 0: 
        length = 0
        offset = 0
        for i in range(1, len(lookahead_buffer) + 1): 
            found = False
            for j in range(len(search_buffer)-i, -1, -1): 
                if lookahead_buffer[:i] == search_buffer[j:j+i]:
                    length = i
                    offset = len(search_buffer)-j
                    found = True
                    break
            if found == False: 
                break
        if length < len(lookahead_buffer): 
            next_chr = lookahead_buffer[length]
        else: 
            next_chr = None
        #print(search_buffer, lookahead_buffer)
        search_buffer.extend(lookahead_buffer[:length+1])
        lookahead_buffer = lookahead_buffer[length+1:]
        if len(string) > 0:
            lookahead_buffer.extend(string[:min(window_size, len(string))])
            string = string[min(window_size, len(string)):]
        result.append((offset, length, next_chr))
    return result

def decompress(compressed: list) -> str:
    output = []
    for offset, length, next_chr in compressed: 
        start = len(output) - offset
        output.extend(output[start: start+length])
        if next_chr is not None: 
            output.append(next_chr)
    return ''.join(output)

string = "you've got to dig it to dig it, you dig?"
print(string)
print(z := compress(string, 100))
print(new_string := decompress(z))
print("Lossless!" if string == new_string else "Lossy!")
