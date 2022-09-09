# sprite
Spelling correction

Pretrained model

phRase mining

Info extraction

Toxic detection

Embedding

## Introduction
`sprite` is a pipeline for text processing.

## Concepts

### spelling correction
Spelling correction module takes care of correcting miss spelled words.

```python
from sprite import MLPCorrector

c = MLPCorrector()
c.from_pretrained('mlp_corrector')

text = 'i like coding , but i hate plya fre frie ! FPS games aer boring'
result = c.correct(text)
print(result)
# "i like coding , but i hate play free fire ! fps games are boring"
text = ['i like coding , but i hate plya fre frie !',
'FPS games aer boring',]
result = c.batch_correct(text,num_workers=2)
print(result)
```

### tokenizations
Tokenizations module takes care of encoding text into model-readble types.

```python
from sprite import BertTokenizer

t = BertTokenizer()
t.from_pretrained('bert_tokenizer')

text = 'i like free fire!'
t.tokenize(text)
t.encode(text)

```

### pretrained models
This module takes care of storing pretrained models for fine tuning.

```python
from sprite import BERT
from sprite import BertTokenizer
import torch

text = 'i like free fire!'

t = BertTokenizer()
t.from_pretrained('bert_tokenizer')
model = BERT()
model.from_pretrained('bert_51m')
model.eval()

input = torch.tensor(t.encode(text)).view(1,-1)
out = model(input)
```

### toxic detection
This module takes care of detecting toxic contents of a given text.

```python
from sprite import BertToxicDetector

model = BertToxicDetector()
model.from_pretrained("toxic_detector_bert_51m")

model.detect("i like free fire.")
```
