import transformers, torch
from abc import ABC, abstractmethod

class LMWrapper(ABC):

    @property
    @abstractmethod
    def ALPHABET_SIZE() -> int:
        pass

    @property
    @abstractmethod
    def BOS_TOKEN() -> int:
        pass

    @abstractmethod
    def tokenize(condition: torch.Tensor): 
        pass

    @abstractmethod
    def detokenize(condition: torch.Tensor): 
        pass
    
    @abstractmethod
    def batch_p_given(condition: torch.Tensor) -> torch.Tensor: 
        pass

    @abstractmethod
    def p_given(condition: torch.Tensor, kv_cache: torch.Tensor = None) -> torch.Tensor: 
        pass


class GPT2(LMWrapper): 
    
    def __init__(self):
        self.model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.model.eval()
    
    @property
    def BOS_TOKEN(self): 
        return self.tokenizer.bos_token_id
    
    @property
    def ALPHABET_SIZE(self):
        return 50257
    
    def tokenize(self, inp: str) -> torch.Tensor:
        return self.tokenizer.encode(inp.read().decode("utf-8"), return_tensors="pt")[0]
    
    def detokenize(self, inp: int) -> str:
        return self.tokenizer.decode(inp)
    
    def batch_p_given(self, condition: torch.Tensor) -> torch.Tensor:
        # Condition is a tensor of shape (sequence length, )
        input_ids = condition.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids)
        
        prob = torch.softmax(outputs.logits.squeeze(0), dim=-1)
        return prob

    def p_given(self, input_id, kv_cache=None) -> torch.Tensor:
        input_id = input_id.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_id, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values

        prob = torch.softmax(outputs.logits[0, -1], dim=-1)

        return prob, kv_cache


class Llama3(LMWrapper): 
    
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        self.model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
        self.model.eval()
    
    @property
    def BOS_TOKEN(self): 
        return 128000
    
    @property
    def ALPHABET_SIZE(self):
        return 128256
    
    def tokenize(self, inp: str) -> torch.Tensor:
        return self.tokenizer.encode(inp.read().decode("utf-8"), return_tensors="pt")[0]
    
    def detokenize(self, inp: int) -> str:
        return self.tokenizer.decode(inp)
    
    def batch_p_given(self, condition: torch.Tensor) -> torch.Tensor:
        # Condition is a tensor of shape (sequence length, )
        input_ids = condition.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids)
        
        prob = torch.softmax(outputs.logits.squeeze(0), dim=-1)
        return prob

    def p_given(self, input_id, kv_cache=None) -> torch.Tensor:
        input_id = input_id.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_id, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values

        prob = torch.softmax(outputs.logits[0, -1], dim=-1)

        return prob, kv_cache