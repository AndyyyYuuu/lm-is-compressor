from collections.abc import Callable
import torch, numpy
from modelling.architecture import Architecture

class Distribution: 
    def __init__(self, p: Callable, alphabet: list): 
        self.p = p
        self.alphabet = alphabet


class DistFromModel: 
    def __init__(self, path: str, temperature: float = 0.8): 
        model_params, self.char_to_int = torch.load(path)
        self.int_to_char = dict((i, c) for c, i in self.char_to_int.items())
        self.alphabet = list(self.char_to_int.keys())
        self.model = Architecture(len(self.alphabet))
        self.model.load_state_dict(model_params)
        self.model.eval()
        self.temperature = temperature
        self.end_prob = 0.001
    
    def get_alphabet(self): 
        return self.alphabet + ["<END>"]

    
    def p(self, item: str, seq: list=[]) -> float:
        if item == "<END>": 
            return self.end_prob
        if len(seq) == 0:
            return 1 / (len(self.alphabet)+1)  # for now, assume uniform distribution of first item
        if item not in self.alphabet: 
            raise ValueError(f"Character {item} not in alphabet")
        seq = [self.char_to_int[i] for i in seq]
        x = numpy.reshape(seq, (1, len(seq), 1)) / float(len(self.alphabet))
        x = torch.tensor(x, dtype=torch.float32)
        prediction = self.model(x)
        prediction_probs = torch.softmax(prediction/self.temperature, dim=1)
        prediction_probs = prediction_probs.squeeze().detach().numpy() * (1-self.end_prob)
        return float(prediction_probs[self.char_to_int[item]])
    
    def __repr__(self):
        return f"DistFromModel( \n\talphabet: {self.alphabet} \n\t\tsize: {len(self.alphabet)}\n)"

d = DistFromModel("modelling/hemingway.pth")
print(d)
print(d.p("r", ["h","e"]))