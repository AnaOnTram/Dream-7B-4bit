# Model Info
[![Static Badge](https://img.shields.io/badge/Hugging%20Face%20🤗-Dream%207B_Base-blue)
]([https://huggingface.co/Dream-org/Dream-v0-Base-7B](https://huggingface.co/Rainnighttram/Dream-v0-Instruct-7B-4bit))
[Rinnighttram/Dream-v0-Instruct](https://huggingface.co/Rainnighttram/Dream-v0-Instruct-7B-4bit)
4 bit quantization of [HKUNLP/Dream](https://github.com/HKUNLP/Dream) Dream-7B Diffusion Language Model.
## Syetm Requirements
- Over 10 GB Free Storage
- Recommend to have over 10GB VRAM
- Nvidia Graphic Card (Need bitsandbytes support)
## Usage
Clone the repository 
```bash
git clone https://github.com/AnaOnTram/Dream-7B-4bit.git
```

## pre-requirements
```bash
conda create -n Dream python=3.11
pip install transformers==4.46.2 torch==2.5.1 bitsandbytes accelerate
```

## Demo
For single chat, please run 
```bash
python single_chat.py
```
For multi-rounds chat, please run 
```bash
python multi_chat.py
```
## Notice
The model was quantized on a single RTX A6000 Machine using bitsandbytes. Compared with original model published by HKUNLP (which requires over 20GB VRAM), the 4bit quantized model could run on a single RTX3080. With normal context length, the model consumes around 9.5GB VRAM and took significantly longer time to generate outputs.
