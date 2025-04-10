from transformers import AutoModel, AutoTokenizer

model_path = "Rainnighttram/Dream-v0-Instruct-7B-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
     model_path,
     device_map="auto",
     trust_remote_code=True
)
model = model.to("cuda").eval()

messages = [
     {"role": "user", "content": "Please make comparisons between UHF and LF RFID."}
 ]

inputs = tokenizer.apply_chat_template(
     messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
 )
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
     input_ids,
     attention_mask=attention_mask,
     max_new_tokens=512,
     output_history=True,
     return_dict_in_generate=True,
     steps=512,
     temperature=0.2,
     top_p=0.95,
     alg="entropy",
     alg_temp=0.,
 )

generations = [
     tokenizer.decode(g[len(p) :].tolist())
     for p, g in zip(input_ids, output.sequences)
 ]

print(generations[0].split(tokenizer.eos_token)[0])
