## Environment
``` bash
if [ -d /data/v-pezhang/nn/lib/python3.8/site-packages/torch ];
then
  git config --global user.name 'namespace-Pt'
  git config --global user.email 'zpt@ruc.edu.cn'
  sudo apt-get install screen
  conda init
  echo 'alias nn="conda activate /data/v-pezhang/nn"' >> ~/.bashrc
  source ~/.bashrc
  nn

  code -r /data/v-pezhang/Workspaces/Research.code-workspace
  sleep infinity

else
  git config --global user.name 'namespace-Pt'
  git config --global user.email 'zpt@ruc.edu.cn'

  conda create --prefix /data/v-pezhang/nn python=3.8 -y
  conda init
  echo 'alias nn="conda activate /data/v-pezhang/nn"' >> ~/.bashrc
  source ~/.bashrc
  nn

  ipython kernel install --name "nn" --user

  pip install torch+cu110 -f https://download.pytorch.org/whl/torch_stable.html
  pip install tensorboard ipython jupyter notebook typing pandas scipy scikit-learn transformers

  cd /data/
  mkdir v-pezhang
  cd v-pezhang
  mkdir Data
  mkdir Code
  mkdir Workspace
  cd Code
  git clone https://github.com/namespace-Pt/Document-Reduction.git

  sudo apt-get install rsync
  sudo apt-get install screen
  sleep infinity
fi

```
- `python==3.8`
- `torch==1.7.1`

## Instruction
```bash
cd /data/v-pezhang/Code/Document-Reduction/Code
python -m scripts.esm -m tune -s large -bs=25 -ws=2
python -m scripts.ttm -m tune -s large -encn=bert -sl=30 -bs=25 -is=10 --no_dedup -ws=2

python -m scripts.esm -m tune -s demo -bs=5 -is=10 -d=1
python -m scripts.ttm -m tune -s demo -bs=5 -is=10 -d=1 -encn=bert -sl=30
python -m scripts.ttms -m tune -s demo -bs=5 -is=10
python -m scripts.sfi -m tune -s demo -k=5 -sl=20 -bs=10 -is=10 --no_dedup

python -m scripts.ttms -m inspect -s demo -ck=4123
```
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
- [ ] LSTUR
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
- [ ] differentiable and efficient
  - basically impossible to implement

## Interactor
- [ ] Bert
  - [x] order embedding across the whole user history
  - [x] cls pooling
  - [x] one-pass bert
  - [x] personalized terms do not interact with each other?
  - [x] candidate news attention mask
  - [x] insert CLS in embedding layer, not fusion
    - fobiddable, otherwise the [CLS] would be extracted
  - [x] position embedding in interactor
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

## Questions
- use bag-of-words to encode user history?
  - currently yes
- remove stop words like "-"?
- focus on which task?
  - two tower with bow?
  - two tower with sequence input?

## Issue
- the gradient after docReducer is sharp
- if we concat title with abstract, then there must be many duplicated words, how about removing them when preprocessing?
- learning rate of bert
- use selected document for final bert is terrible
- **history in MIND, the more recent is at the front or the tail**, we now use the head as the latest news by default
  - its ascending by time
- [SEP] no position embedding
- directly using [CLS] as user profile in TTMS results in high loss, and the model can learn nothing
- **one tower is not better than two tower, why?**

## Ablation
- Two Tower
  - [ ] encode user representation from news/bag-of-words
  - user representation as aggregation of historical news
    - [69.93] signal_length=30
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
  - user representation as [CLS] of historical news sequence
    - [ ] signal_length=5
    - [ ] bm25 top 5
    - selected top 5
      - [ ] non-mask, non-deduplicate, non-BoW
      - [ ] +mask
      - [ ] +deduplicate
      - [ ] +BoW

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
## Function
### Embedding
### Encoder
- output no levels

### DocReducer
- no level