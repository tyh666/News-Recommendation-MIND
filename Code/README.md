## Environment
``` bash
if [ -d /data/v-pezhang/nn/lib/python3.8/site-packages/torchtext ];
then
  git config --global user.name 'namespace-Pt'
  git config --global user.email 'zpt@ruc.edu.cn'
  sudo apt-get install screen
  conda init
  echo 'alias nn="conda activate /data/v-pezhang/nn"' >> ~/.bashrc
  source ~/.bashrc
  nn
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

  pip install torch==1.7.1+cu110 torchtext==0.8.1 pandas scipy scikit-learn transformers -f https://download.pytorch.org/whl/torch_stable.html
  pip install tensorboard ipython jupyter notebook typing

  cd /data/
  mkdir v-pezhang
  cd v-pezhang
  mkdir Data
  mkdir Code
  cd Code
  git clone https://github.com/namespace-Pt/Document-Reduction.git

  sudo apt-get install rsync
  sudo apt-get install screen
  sleep infinity

fi
```
- `python==3.8`
- `torch==1.7.1`
- `torchtext == 0.8.1`

## Instruction
```bash
cd /data/v-pezhang/Code/Document-Reduction/Code
python esm.py -m tune -s demo -emb=bert -lr=1e-3 -bs=25 -is=50 -ws=2
```
## Preprocess
- [x] remove stop words
  - not necessary
- [ ] check the average length of title+abstract+vert
  - [ ] propotion > 200
  - [ ] propotion > 300
  - [ ] propotion > 400

## Embedding
- [x] Random embedding
- [ ] Bert embedding
  - [x] add absolute position encoding
  - [x] add cls embedding

## News Encoder
- [x] CNN encoder
- [x] FIM encoder
- [ ] Bert encoder
  - [ ] [CLS] as news repr
  - [ ] attention over the last layer as news repr
- [ ] **how to deal with the categories**

## User Encoder
- [x] RNN encoder
- [ ] LSTUR
- [ ] Adaptive and time-aware LSTUR, see [29]
- [ ] NRMS

## Document Reducer
- [ ] matching based
  - [x] only topk
  - [ ] topk with threshold
  - [ ] diversified
    - [ ] use document vector as an extraction input
    - [ ] design methods to extract non-duplicated terms within an article
  - [ ] extract origin embedding, not after cnn
  - [ ] dynamic allocation of top k
    - [ ] read paper of yi xiaoyuan
- [ ] Seq2Seq based
- [ ] RL based

## Term Fuser
- [ ] differentiable and efficient
  - basically impossible to implement

## Interactor
- [ ] Bert
  - [x] position embedding within each news
  - [x] order embedding across the whole user history
  - [x] cls pooling
  - [ ] one-pass bert
    - [ ] relative position embedding?
  - [ ] personalized terms do not interact with each other?
  - [x] candidate news attention mask
  - [x] insert CLS in embedding layer, not fusion
    - fobiddable, otherwise the [CLS] would be extracted
  - [x] position embedding in interactor
  - [x] one entry to define term_num
  - [ ] speed compare between onepass bert and overlook bert
  - [ ] 

## Workflow
- [x] extract terms from every historical news when the history is updated
- [ ] extract terms incrementally
- [ ] long and short term extraction
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
- [ ] manager.encode
- [x] distributed evaluate

## Issue
- the gradient after docReducer is sharp
- if we concat title with abstract, then there must be many duplicated words, how about removing them when preprocessing?
- very likely to fall into local optim
- learning rate of bert

## Need to update
- [ ] Encoders.MHA, NPA, Pipeline

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
- map dataset for MIND
- cache enabled

## Function
### Embedding
### Encoder
- may have levels
- output no levels

### DocReducer
- no level