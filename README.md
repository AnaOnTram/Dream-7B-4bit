# Model Info
4 bit quantization of [HKUNLP/Dream](https://github.com/HKUNLP/Dream) Dream-7B Diffusion Language Model.
# Syetm Requirements
- Over 10 GB Free Storage
- Recommend to have over 10GB VRAM
- Nvidia Graphic Card (Need bitsandbytes support)
# Usage
Clone the repository ```bash
git clone https://github.com/AnaOnTram/Dream-7B-4bit.git
```
##pre-requirements
```bash
conda create -n Dream python=3.11
pip install transformers==4.46.2 torch==2.5.1 bitsandbytes accelerate
```
##Demo
For single chat please run ```bash
python single_chat.py
```
For multi-rounds chat, please run  ```bash
python multi_chat.py
```
