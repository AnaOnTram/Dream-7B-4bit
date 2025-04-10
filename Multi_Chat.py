from transformers import AutoModel, AutoTokenizer

def initialize_model():
    model_path = "Rainnighttram/Dream-v0-Instruct-7B-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    model = model.to("cuda").eval()
    return model, tokenizer

def generate_response(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        return_dict=True, 
        add_generation_prompt=True
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
        tokenizer.decode(g[len(p):].tolist())
        for p, g in zip(input_ids, output.sequences)
    ]

    return generations[0].split(tokenizer.eos_token)[0]

def main():
    print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model()
    
    messages = []
    
    print("Chat initialized. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nEnding conversation. Goodbye!")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="")
        response = generate_response(model, tokenizer, messages)
        print(response)
        
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
