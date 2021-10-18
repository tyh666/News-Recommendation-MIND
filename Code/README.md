## Environment
``` bash
if [ -d /data/v-pezhang/nn/lib/python3.8/site-packages/torch ];
then
  git config --global user.name 'namespace-Pt'
  git config --global user.email 'zpt@ruc.edu.cn'
  sudo apt-get install screen -y
  sudo apt-get install rsync -y
  conda init
  if ! [ -n nn ]; then
    echo 'alias nn="conda activate /data/v-pezhang/nn"' >> ~/.bashrc
  fi

  if ! [ -n pt ]; then
    echo 'alias pt="screen -r -d pt"' >> ~/.bashrc
  fi
  source ~/.bashrc
  nn

  /data/v-pezhang/nn/bin/pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

  sleep infinity

else
  git config --global user.name 'namespace-Pt'
  git config --global user.email 'zpt@ruc.edu.cn'

  conda create --prefix /data/v-pezhang/nn python=3.8 -y
  conda init
  echo 'alias nn="conda activate /data/v-pezhang/nn"' >> ~/.bashrc
  source ~/.bashrc
  nn
  /data/v-pezhang/nn/bin/pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  /data/v-pezhang/nn/bin/pip install tensorboard ipython jupyter notebook typing pandas scipy scikit-learn transformers

  cd /data/
  mkdir v-pezhang
  cd v-pezhang
  mkdir Data
  mkdir Code
  cd Code
  git clone https://github.com/namespace-Pt/Document-Reduction.git

  sudo apt-get install rsync -y
  sudo apt-get install screen -y
  sleep infinity
fi
```

- `python==3.8`
- `torch==1.7.1`

## Instruction
```bash
cd /data/v-pezhang/Code/Document-Reduction/Code
python -m scripts.esm -m train -s large -ws=2
python -m scripts.ttm -m train -s large -red=none -ws=2
python -m scripts.tesrec -m train -s large -ws=2
python -m scripts.plm -m train -s large -red=none -ws=2

python -m scripts.esm -m train -s demo -bs=5 -is=10 -d=1
python -m scripts.ttm -m train -s demo -bs=5 -is=10 -red=none -d=2
python -m scripts.plm -m train -s demo -bs=5 -is=10 -red=none -d=2
python -m scripts.tesrec -m train -s demo -bs=5 -is=10
python -m scripts.sfi -m train -s demo -k=5 -is=10 -sl=20 -bs=10 --no_dedup

python -m scripts.tesrec -m inspect -s large -ck=60000 -d=-1

python -m scripts.tesrec -m test -s large -ck=589
python -m scripts.esm -m test -s large -ck=150000
```

## TODO
- [ ] Do I need to compare a text summarization model?
- [ ] whole dataset

## Preprocess
- [x] remove stop words
  - not necessary
- [ ] **currently we append the topic and subtopic at the tail of document, maybe at the front may be better?**
- [ ] check the average length of title+abstract+vert
  - [ ] propotion > 200
  - [ ] propotion > 300
  - [ ] propotion > 400
- [x] **bag of words input**

## Embedding
- [x] Random embedding
- [x] Bert embedding
  - [x] add absolute position embedding
  - [x] add cls embedding
  - [x] add token type embedding
    - add in the final ranking stage

## News Encoder
- [x] CNN encoder
- [x] FIM encoder
- [x] Bert encoder
  - [x] [CLS] as news repr
  - [x] attention over the last layer as news repr
    - not necessary, the CLS is already the attentive pooling output

## User Encoder
- [x] RNN encoder
- [x] NRMS
- [x] LSTUR
- [ ] Adaptive and time-aware LSTUR, see [29]

## Document Reducer
- [ ] matching based
  - [x] only topk
  - [x] topk with threshold
  - [x] extract after masking
  - [ ] diversified
    - [x] use document vector as an extraction input
    - [ ] design methods to extract non-duplicated terms within an article
      - [x] attention mask
      - [x] input bag-of-words
  - [x] extract origin embedding, not after cnn
  - [ ] dynamic allocation of top k
    - [x] read paper of yi xiaoyuan
  - [x] extract terms from every historical news when the history is updated
  - [ ] extract terms incrementally
  - [ ] long and short term extraction
  - [x] **modify his_attn_mask to excludes repetitve terms from selection**

- [ ] Seq2Seq based
- [ ] RL based

## Term Fuser
- [x] differentiable and efficient
  - basically impossible to implement

## Ranker
- [ ] Bert
  - [x] order embedding across the whole user history
  - [x] cls pooling
  - [x] one-pass bert
  - [x] personalized terms do not interact with each other?
  - [x] candidate news attention mask
  - [x] insert CLS in embedding layer, not fusion
    - fobiddable, otherwise the [CLS] would be extracted
  - [x] position embedding in Ranker
  - [x] one entry to define term_num
  - [x] speed compare between onepass bert and overlook bert
    - onepass is faster when training
    - overlook is faster when evaluating
  - [x] recent K

## Workflow
- [x] auto-regressive user modeling
- [x] distributed training
- [x] **config.filter_num -> config.hidden_dim**
- [x] **config.signal_length**
- [x] **config.embedding_dim**
- [x] modify load_config
- [x] cache dataset
  - [x] refactor dataset: encode all tokens, and dispatch/truncate when getting items
- [x] integrate manager and config
- [x] distributed evaluate
- [ ] customized docker image
- [x] prepare modify dataset.vocab
- [ ] specify .vscode-sever location to avoid repeatedly downloading extensions
- [x] manager.encode
- [x] distributed evaluate
- [x] reverse history when learning user profile
- [x] which part consumes time most?

## Experiments
- [ ] no order embedding, no scheduler, token level
- [ ]

## Questions
- use bag-of-words to encode user history?
  - currently yes
- remove stop words like "-" in bow?
- focus on which task?
  - two tower with bow?
  - two tower with sequence input?
- can only select tokens, rather than words
- padded news: [CLS] [SEP] [PAD] [PAD] ... or [PAD] [PAD] ...

1. Performance on dev is far better than test (71.8 vs 69.76)
2. Can I write manuscipt now? Why should we select personalized terms?
3. Do I need to rerun fastformers?

## Issue
- the gradient after docReducer is sharp
- if we concat title with abstract, then there must be many duplicated words, how about removing them when preprocessing?
- learning rate of bert
- use selected document for final bert is terrible
- **history in MIND, the more recent is at the front or the tail**, we now use the head as the latest news by default
  - its ascending by time
- **one tower is not better than two tower, why?**

## Ablation
- Two Tower
  - [ ] encode user representation from news/bag-of-words
  - [69.93] signal_length=30

  - user representation as [CLS] of historical news sequence
    - [ ] signal_length=5
    - [69.59] bm25 top 5
    - selected top 5
      - [ ] non-mask, non-deduplicate, non-BoW
      - [ ] +mask
      - [71.21] +deduplicate
      - [70.9] +BoW

  - user representation as aggregation of historical news
    - [ ] bm25 top 30
    - selected top 5
      - [ ] non-mask, non-deduplicate, non-BoW
      - [ ] +mask
      - [ ] +deduplicate
      - [ ] +BoW
    - [ ] signal_length=80
      - [ ] speedyfeed
    - user encoder
      - [ ] RNN
      - [ ] CNN
      - [ ] MHA

- One Tower
  - [ ] signal_length=5
  - [ ] bm25 top 5
  - [69.91] selected top 5

## Need to update
- [x] Encoders.MHA,
- [ ] NPA
- [ ] Pipeline

## Phylosiphy
### manager
- a class
- the attributes are hyper parameters
- the function wraps logging/training/evaluating process for the model
### model
- nn.Module
- posses all necessary (used in inference) hyper parameters as attributes
- posses some unique attributes per model (name, term_num)
### MIND
map dataset for MIND
- `input`
- `output`:
  - *_encoded_index: tokenized news
  - *_reduced_index: reduced tokenized news
  - *_attn_mask: attention mask
  - *_reduced_mask: reduced attention mask

##### Reducers
- `matching`:
  - `encoded_news`: news tokens of orignial text, truncated to `signal_length`
  - `attn_mask`: original attention mask, truncated to `signal_length`
  - `attn_mask_refined`: deduplicated attention mask, if `no_dedup=False(default)`
- `bm25`:
  - `encoded_news`: news tokens of descendingly sorted words according to bm25 scores, truncated to `signal_length`
  - `encoded_news_original`: news tokens of orignial text, truncated to `signal_length`
  - `attn_mask`: attention mask of bm25 tokens, truncated to `signal_length`
  - `attn_mask_original`: original attention mask, truncated to `signal_length`
- `bow`:
  - `encoded_news`: news token-count pairs, truncated to `signal_length`
  - `attn_mask`: attention mask of token-count pairs


## Function
### Embedding
### Encoder
- output no levels

### DocReducer
- no level