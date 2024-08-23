import torch
from pathlib import Path
import numpy as np
import json
import os
import glob
import torch.nn.functional as F
from litgpt import LLM
from litgpt.data import JSON
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to calculate the running average (moving average)
def running_average(data, window_size):
    import pandas as pd
    return pd.Series(data).rolling(window=window_size).mean().to_numpy()
# max_seq_length = 8192
max_seq_length = 2048
# llm = LLM.load("microsoft/phi-2")
# llm = LLM.load("meta-llama/Meta-Llama-3.1-8B")
# llm = LLM.load("out/finetune/lora-llama-3.1-8b-speech/final/")
llm = LLM.load("out/finetune/lora-llama-3.1-8b-instruct-speech-2k/final")
tokenizer = llm.preprocessor.tokenizer
fabric = llm.fabric

data = JSON(json_path=Path("data/crawl_from_youtube/speech_sft.json"), val_split_fraction=0.2, mask_prompt=False, prompt_style="llama3")
data.connect(tokenizer=tokenizer, batch_size=1, max_seq_length=max_seq_length)
data.prepare_data()
data.setup()
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()
train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

with torch.no_grad():
    sum_loss = torch.zeros(max_seq_length-1, device=llm.fabric.device)
    valid_count = torch.zeros(max_seq_length-1, device=llm.fabric.device)
    # for batch in tqdm(val_dataloader):
    for batch in tqdm(train_dataloader):
        input_ids, targets = batch["input_ids"], batch["labels"]
        print('input_ids:', input_ids.shape)

        logits = llm.model(input_ids)

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits[..., :-1, :], targets[..., 1:], reduction='none')
        sum_loss[:len(loss)] += loss
        valid_count[:len(loss)] += 1

# avg_ppl = (sum_loss / valid_count).exp().cpu().numpy()
avg_loss = (sum_loss / valid_count).cpu().numpy()
avg_loss = running_average(avg_loss, 10)
avg_ppl = np.exp(avg_loss)


# save the loss to a file
plt.plot(avg_ppl)
plt.xscale('log')
plt.yscale('log')
# plt.xticks([256, 512, 1024, 2048, 4096, 8192], labels=['256', '512', '1k', '2k', '4k', '8k'])
plt.xlim(128)
plt.minorticks_off()
plt.xticks([128, 256, 512, 1024, 2048], labels=['128', '256', '512', '1k', '2k'])
plt.yticks([10, 9.5, 9.0, 8.5])
plt.xlabel('Position')
plt.ylabel('Average Perplexity')
plt.title('Average Perplexity at Each Position')
plt.savefig('avg_ppl_3.png')
plt.clf()

plt.plot(avg_loss)
plt.xlabel('Position')
plt.ylabel('Average Loss')
plt.xscale('log')
plt.xlim(128)
plt.minorticks_off()
# plt.xticks([256, 512, 1024, 2048, 4096, 8192], labels=['256', '512', '1k', '2k', '4k', '8k'])
plt.xticks([128, 256, 512, 1024, 2048], labels=['128', '256', '512', '1k', '2k'])
plt.title('Average Loss at Each Position')
# plt.savefig('avg_loss_1.png')
plt.savefig('avg_loss_3.png')


# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker

# import numpy as np
# from scipy.interpolate import interp1d


# window = 10
# all_x_labels = ['128', '256', '512', '1k', '2k']
# all_x_vals = 2**(7+np.asarray(range(len(all_x_labels))))

# def helper(data):

#     x_lin = np.asarray(range(len(data))) + 1
#     f_interp = interp1d(x_lin, data, kind='linear', fill_value='extrapolate')

#     x_data = np.logspace(0, np.log10(2**11), 1000)
#     data = f_interp(x_data)

#     if window > 0:
#         data = np.convolve(data, np.ones(window)/window, mode='valid')
#         x_data = x_data = np.logspace(0, np.log10(2**11), len(data))
#         data = data[300:]
#         x_data = x_data[300:]

#     return x_data, data


# plt.figure(figsize=(10, 6))
# x_data, data = helper(avg_loss)
# plt.plot(x_data, data)

# plt.xscale('log')

# plt.tick_params(
#         axis='x',          # changes apply to the x-axis
#         which='minor',     # only minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False) # labels along the bottom edge are off

# plt.xticks(ticks=all_x_vals, 
#            labels=all_x_labels)

# plt.xlabel('Token index in context')
# plt.ylabel('Per-token val loss')
# plt.legend()

# plt.savefig('token.pdf')
# # plt.show()