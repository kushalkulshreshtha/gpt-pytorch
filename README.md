A replica of GPT-1 model by OpenAI trained using PyTorch. It is a decoder-only transformer model trained on Shakespeare's text at a character token level.

**Files**

[input.txt](data/input.txt): Input text data 

[requirements.txt](requirements.txt): Envoronment setup

[train.py](train.py): Code containing the training loop in PyTorch

[train_lighning.py](train_lighning.py): Traning loop in Lightning

[model.pt](model/model.pt): Trained GPT Model 

[model_output.txt](model_output.txt): Model output in txt format

**Credits**

Andrej Karpathy's NanoGPT: [NanoGPT](https://github.com/karpathy/nanoGPT)

**Colab Notebook**

Vanilla Pytorch: https://colab.research.google.com/drive/1O4K0hwt5TUg0jiU-i0GH0NV5G-MNEh1k?usp=sharing

Pytorch + Lighning (trained only for 1000 iterations, so output is not so good): https://colab.research.google.com/drive/1gApQGXmqdJ7oiZmRYfxzIMmMOyK0xcOL?usp=sharing
