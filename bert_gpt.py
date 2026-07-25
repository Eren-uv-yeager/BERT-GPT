# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import torch



# model_name="bert-large-uncased-whole-word-masking-finetuned-squad"
# Tokenizer=AutoTokenizer.from_pretrained(model_name)
# model=AutoModelForQuestionAnswering.from_pretrained(model_name)
# if model_name:
#     print("Bert Model loaded successfull")

# question="Who is apple ceo?"
# context="Apple ceo is may not be Mike tyson and steve"

# encoding=Tokenizer(question,context,return_tensors="pt")
# print("Token ids")
# print(encoding["input_ids"])
# print(len(encoding["input_ids"][0]))
# token=Tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
# print(token)
# print(len(token))
# print("Attention mask")
# print(encoding["attention_mask"])
# with torch.no_grad():
#     output=model(**encoding)
# print(output)
# start_index = torch.argmax(output.start_logits)
# end_index = torch.argmax(output.end_logits)

# print("start_index",start_index)
# print("end_index",end_index)
# print(start_index.item())
# print(end_index.item())
# answer_ids = encoding["input_ids"][0][start_index:end_index+1]
# answer = Tokenizer.decode(answer_ids, skip_special_tokens=True)
# print(answer)

# def question_from_bert():
#     if answer and question:
#         return [question,answer]
# print(question_from_bert())
from model import Bert_Gpt
question="Who Build taj Mahal"
context="Taj mahal has been built by Shajahan"
model_name="bert-large-uncased-whole-word-masking-finetuned-squad"
bert=Bert_Gpt()
model=bert.bert_model_token(model_name=model_name)
print("Iam Model",model)
encode=bert.encoding(question=question,context=context)
op=bert.output(encoding=encode)
print("Output",op)
start=bert.get_start_logits(op)
end=bert.get_end_logits(op)
print("start_index",start,"\n","end_index",end)
index_in_context=bert.get_answer(encoding=encode,start_index=start,end_index=end)
print("Get index that is start and end in context:",index_in_context)
decode_answer=bert.decode_answer(index_in_context)
print("The answer is",decode_answer)
model_name="gpt2"
bert.gpt_model_token(model_name=model_name)
i_p=i_p = f"Question: {question}\nAnswer: {decode_answer}\nExplain:"
op=bert.generate(i_p,max_new_tokens=50,inference_type="greedy")
print(op)
# encoded_ip=bert.get_encoded_input(i_p=i_p)
# print("iam encoder op from gpt",encoded_ip)
# op=bert.get_gpt_output(encoded_input=encoded_ip)
# logit=bert.get_logits(op)
# last_token_logit=bert.last_token_logits(logits=logit)
# print(last_token_logit)
# print("shape is",last_token_logit.shape)
# soft_max_logit=bert.convert_last_logits_voc_softmax(last_logits=last_token_logit)
# print("soft_max_op",soft_max_logit)
# High_proba_word=bert.get_high_prob_voc(soft_max_value=soft_max_logit)
# print("high_prob word",High_proba_word.item())
# gpt_op=bert.decode_into_word(High_proba_word)
# print(f"Question: {question} ? {decode_answer} ",gpt_op)


