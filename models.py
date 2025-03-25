import transformers, torch
from abc import ABC, abstractmethod

class LMWrapper(ABC):

    @property
    @abstractmethod
    def VOCAB_SIZE() -> int:
        pass

    @property
    @abstractmethod
    def BOS_TOKEN() -> int:
        pass

    @abstractmethod
    def tokenize(condition: torch.Tensor, kv_cache=None: torch.Tensor): 
        pass
    
    @abstractmethod
    def batch_p_given(condition: torch.Tensor) -> torch.Tensor: 
        pass

    @abstractmethod
    def p_given(condition: torch.Tensor, kv_cache=None: torch.Tensor) -> torch.Tensor, torch.Tensor: 
        pass


class GPT2(LMWrapper): 
    
    def __init__():
        self.model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.model.eval()
    
    @property
    def BOS_TOKEN(): 
        return self.tokenizer.bos_token_id
    
    @property
    def VOCAB_SIZE():
        return 50257
    
    def batch_p_given(condition: torch.Tensor) -> torch.Tensor:
        # Condition is a tensor of shape (sequence length, )
        input_ids = condition.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids)
        
        prob = torch.softmax(outputs.logits.squeeze(0), dim=-1)
        return prob

    def p_given(input_id, kv_cache=None) -> torch.Tensor, torch.Tensor:
        input_id = input_id.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_id, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values

        token_log_prob = torch.softmax(outputs.logits[0, -1], dim=-1)

        return prob, kv_cache

