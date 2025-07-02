import torch
from sys import maxunicode
import torch.nn as nn
from torch.nn import functional as F

#-------
blockSize = 8
batchSize = 4
maxIters = 3000
evalInterval = 300
lr = 1e-2
evalIters = 200
#-------
torch.manual_seed(1337)

with open('input.txt', 'r' ,  encoding='utf-8') as f:
  text = f.read()
print(text[:1000])
uniqueChars = sorted(list(set(text))) ## A set of all unique characters that can be recognised or output by the model
vocabSize = len(uniqueChars)

##Tokenises characters by assigning the value of a char by its corresponding position in the UniqueChars list
StringToInt = {ch:i for i,ch in enumerate(uniqueChars)}
IntToString = {i:ch for i,ch in enumerate(uniqueChars)}
encode = lambda x: [StringToInt[c] for c in x]
decode = lambda x: ''.join(IntToString[c] for c in x)

##Encodes the data into a tensor
data = torch.tensor(encode(text),  dtype=torch.long)
##Splits the data into training and evaluation
split = int(0.9*len(data))
trainData = data[:split]
evalData = data[split:]

@torch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ['train' , 'eval']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X,Y = getBatch(split)
            logits, loss = model(X,Y)
            losses[k] = losses.mean()
        model.train()
    return out 

## Y is the targets and X is the context
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
    #print(f"When input is {context}, target is {target}")

class BigramLanguageModel(nn.Module):
  def __init__(self, vocabSize):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocabSize, vocabSize)
  def forward(self, index, targets=None):
    ## index and targets are both batch and time tensors of integers
    logits = self.token_embedding_table(index) ## (batch, time, channel) tensor
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
      logits, loss = self(index)
      logits = logits[:,-1,:] ## Becomes (B,C)
      probDistribution = F.softmax(logits, dim=1) ##(B,C)
      indexNext = torch.multinomial(probDistribution, num_samples=1) ## (B,1), Samples from distribution. Single prediction for each batch hence (B,1)
      index = torch.cat((index, indexNext), dim=1) ## (B, T+1), Appends to text
    return index


model = BigramLanguageModel(vocabSize)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

## Create optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr= 1e-3)

for steps in range(maxIters):
  if(iter % evalInterval==0):
    losses = estimateLoss()
    print(f"step {iter} has loss {losses}")
  xb, yb = getBatch('train')

  logits, loss = model(xb,yb)
  optimiser.zero_grad(set_to_none=True)
  loss.backward()
  optimiser.step()
print(loss.item())

print(decode(model.generate(torch.zeros((1,1),dtype=torch.long), maxNewTokens=100)[0].tolist()))