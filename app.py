from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InferlessPythonModel:
    def initialize(self):
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                inputs=input_ids,
                temperature=0.7,
                max_new_tokens=512,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return {"generated_result": result}

    def finalize(self):
        self.model = None
