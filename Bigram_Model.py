import torch
import torch.nn as nn
from torch.nn import functional as F

#Tutorial from https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4916s

#---------
blockSize = 8
nEmbed = 32
numHeads  =4
headSize = 8
dropout = 0.2
nLayer = 6
batchSize = 4
torch.manual_seed(1337)


#----------

with open('input.txt', 'r' ,  encoding='utf-8') as f:
  text = f.read()
uniqueChars = sorted(list(set(text))) ## A set of all unique characters that can be recognised or output by the model
vocabSize = len(uniqueChars)

StringToInt = {ch:i for i,ch in enumerate(uniqueChars)}
IntToString = {i:ch for i,ch in enumerate(uniqueChars)}
encode = lambda x: [StringToInt[c] for c in x]
decode = lambda x: ''.join(IntToString[c] for c in x)

data = torch.tensor(encode(text),  dtype=torch.long)
##Splits the data into training and evaluation
split = int(0.9*len(data))
trainData = data[:split]
evalData = data[split:]

x = trainData[:blockSize]
y = trainData[1:blockSize+1]
def getBatch(split): ##Splits data into batches
  data = trainData if split == 'train' else evalData
  ix = torch.randint(len(data) - blockSize, (batchSize,))
  x = torch.stack([data[i:i+blockSize] for i in ix])
  y = torch.stack([data[i+1: i+blockSize+1] for i in ix]) ## Offset of x by 1
  return x,y

xb, yb = getBatch('train')

for b in range(batchSize):
  for t in range(blockSize):
    context = xb[b,:t+1]
    target = yb[b,t]
    
## Model

class FeedForward(nn.Module):
  def __init__(self, nEmbed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(nEmbed, 4*nEmbed),
        nn.ReLU(),
        nn.Linear(4*nEmbed, nEmbed),
        nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)


class MultiHeadAttention(nn.Module):
  def __init__(self, numHeads, headSize):
    super().__init__()
    self.heads = nn.ModuleList([Head(nEmbed, headSize) for _ in range(numHeads)])
    self.proj = nn.Linear(nEmbed, nEmbed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = torch.cat([h(x) for h in self.heads], dim=-1)
    x = self.proj(x)
    return x



class Block(nn.Module):
  def __init__(self, nEmbed, numHeads):
    super().__init__()
    headSize = nEmbed // numHeads
    self.sa = MultiHeadAttention(numHeads, headSize)
    self.ffwd = FeedForward(nEmbed)
    self.ln1 = nn.LayerNorm(nEmbed)
    self.ln2 = nn.LayerNorm(nEmbed)

  def forward(self, x):
    clonex = x.clone()
    x = clonex + self.sa(self.ln1(x))
    x = clonex + self.ffwd(self.ln2(x))
    return x

class Head(nn.Module):
  def __init__(self, nEmbed, headSize): # Added nEmbed as input
    super().__init__()
    self.key = nn.Linear(nEmbed, headSize, bias=False) # Use nEmbed as input feature size
    self.query = nn.Linear(nEmbed, headSize, bias=False) # Use nEmbed as input feature size
    self.value = nn.Linear(nEmbed, headSize, bias=False) # Use nEmbed as input feature size
    self.register_buffer('tril', torch.tril(torch.ones(blockSize,blockSize)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    ## Calculate attention scores
    weight = q @ k.transpose(-2,-1)*(headSize**-0.5) # Reverted scaling factor to standard
    weights = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Use self.tril and slice for correct size
    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)
    ##Weighted aggregation
    v = self.value(x)
    out = weights @ v # Use weights here
    
    return out


class BigramLanguageModel(nn.Module):
  def __init__(self, vocabSize):
    super().__init__()
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbed) ## A simple lookup table that stores embeddings of a fixed dictionary and size.
    self.posEmbeddingTable = nn.Embedding(blockSize, nEmbed) ## Positional embedding
    self.blocks = nn.Sequential(
        Block(nEmbed,numHeads=4),
        Block(nEmbed,numHeads=4),
        Block(nEmbed,numHeads=4),
        nn.LayerNorm(nEmbed),
    )
    self.lm_head = nn.Linear(nEmbed, vocabSize)
  def forward(self, index, targets=None):
    ## index and targets are both batch and time tensors of integers

    B, T = index.shape

    tokenEmbeddings = self.tokenEmbeddingTable(index) ## (batch, time, channel) tensor
    
    pos_indices = torch.arange(T, device=index.device) % blockSize
    posEmbeddings = self.posEmbeddingTable(pos_indices) #(T,C)
    x = tokenEmbeddings + posEmbeddings
    x = self.blocks(x)
    logits = self.lm_head(x) #(B,T, Vocabsize)

    if targets is None:
        return logits, None
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, index, maxNewTokens):
    ##index is (B,T) array
    for _ in range (maxNewTokens):
      cropindex = index[:, -blockSize:] if index.shape[1] > blockSize else index # Corrected slicing
      logits, loss = self(cropindex)
      logits = logits[:,-1,:] ## Becomes (B,C)
      probDistribution = F.softmax(logits, dim=1) ##(B,C)
      indexNext = torch.multinomial(probDistribution, num_samples=1) ## (B,1), Samples from distribution. Single prediction for each batch hence (B,1)
      index = torch.cat((index, indexNext), dim=1) ## (B, T+1), Appends to text
    return index


model = BigramLanguageModel(vocabSize)
logits, loss = model(xb, yb)
#print(logits.shape)
print(loss)

## Create optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr= 1e-3)
batchSize = 8
for steps in range(10000):
  xb, yb = getBatch('train')

  logits, loss = model(xb,yb)
  optimiser.zero_grad(set_to_none=True)
  loss.backward()
  optimiser.step()
print(loss.item())

#Generate text
print(decode(model.generate(torch.zeros((1,1),dtype=torch.long), maxNewTokens=500)[0].tolist()))