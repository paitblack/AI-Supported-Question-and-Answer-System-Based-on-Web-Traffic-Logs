import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ResponseGenerator():
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)  #initializes the T5 tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name) #initializes T5 model from pre trained model specified.

    def generate_response(self, question, context, max_length=200, num_beams=8, early_stopping=True):
        input_text = f"question: {question} context: {context}"  # comine question and the context get from FAISS to be answered.
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt") #encode it

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True) #decode it
        return answer