import torch
from gpt import (
    GPTLanguageModel,
    device,
    decode,
    encode,
)

model = GPTLanguageModel()
model.load_state_dict(torch.load("gpt.checkpoint"))
m = model.to(device)

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = torch.tensor([encode("All:")], dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist())
