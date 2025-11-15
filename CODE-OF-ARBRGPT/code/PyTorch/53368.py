import torch

import torch.nn as nn



embeds = nn.Embedding(5, 12, padding_idx=0) # Weights at 0 are initialized with zeros

embeds.weight.data[0] = torch.ones(12) # Put another value

embeds(torch.tensor([0]))