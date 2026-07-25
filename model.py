from transformers import AutoTokenizer, AutoModelForQuestionAnswering , AutoModelForCausalLM
import  torch
class Bert_Gpt:
    def bert_model_token(self,model_name):
        self.model=AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.token=AutoTokenizer.from_pretrained(model_name)
    def encoding(self,question,context):
        return self.token(question,context,return_tensors="pt")
    def output(self,encoding):
        with torch.no_grad():
            output=self.model(**encoding)
            return output
    def get_start_logits(self,output):
        return torch.argmax(output.start_logits)
    def get_end_logits(self,output):
        return torch.argmax(output.end_logits)
    def get_answer(self,encoding,start_index,end_index):
        return encoding["input_ids"][0][start_index:end_index+1]
    def decode_answer(self,answer):
        return self.token.decode(answer, skip_special_tokens=True)
    def gpt_model_token(self,model_name):
        self.gpt_model=AutoModelForCausalLM.from_pretrained(model_name)
        self.gpt_tokenizer=AutoTokenizer.from_pretrained(model_name)
    def generate(self, prompt, max_new_tokens=None,inference_type=None,top_p=None):
        encoded = self.gpt_tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self.gpt_model(input_ids=input_ids,attention_mask=attention_mask)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=1)
            next_token = torch.argmax(probs, dim=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)],dim=1)
            attention_mask = torch.cat([attention_mask,torch.ones((1, 1), dtype=attention_mask.dtype)],dim=1)

        return self.gpt_tokenizer.decode(input_ids[0],skip_special_tokens=True)

# elif inference_type=="top_p":
#                 probs = torch.softmax(logits, dim=-1)
#                 sorted_probs, sorted_indices = torch.sort(probs,descending=True,dim=-1)
#                 cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
#                 mask = cumulative_probs > top_p
#                 mask[..., 1:] = mask[..., :-1].clone()
#                 mask[..., 0] = False
#                 sorted_probs[mask] = 0
#                 sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
#                 next_token_position = torch.multinomial(sorted_probs,num_samples=1)
#                 next_token = torch.gather(sorted_indices,-1,next_token_position)
#                 next_token = next_token.squeeze(-1)