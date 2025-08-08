import torch

from transformers import pipeline


model_id ="Intel/gpt-oss-20b-int4-AutoRound"

import pdb; pdb.set_trace()
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    #revision="60d5a00b7f9295317eb693a437f413503f045de1",
)
print(pipe.model)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1])

