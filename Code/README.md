## Preprocess
- [x] remove stop words
  - not necessary
- [ ] check the average length of title+abstract+vert
  - [ ] propotion > 200
  - [ ] propotion > 300
  - [ ] propotion > 400

## News Encoder
- [x] CNN encoder
  - [ ] output hidden_dim=768
- [ ] FIM encoder
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
  - [ ] dynamic allocation of top k
    - [ ] read paper of yi xiaoyuan
- [ ] Seq2Seq based
- [ ] RL based

## Term Fuser
- [ ] differentiable and efficient

## Interactor
- [ ] Bert
  - [x] position embedding within each news
  - [x] order embedding across the whole user history
  - [ ] pooling
    - [x] pool with cls
    - [ ] pool with mean
- [ ] FIM
- [ ] KNRM

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

## Issue
- the gradient after docReducer is sharp
- if we concat title with abstract, then there must be many duplicated words, how about removing them when preprocessing?

in SpeedyFeed
- industry data for speedyfeed
- still fit on one gpu?

## Need to update
- [ ] Encoders.MHA, NPA, Pipeline, Random Embedding
- [ ]

## Phylosiphy
### manager
- a class
- the attributes are hyper parameters
- the function wraps logging/training/evaluating process for the model
### model
- nn.Module
- posses all necessary (used in inference) hyper parameters as attributes
- posses some unique attributes per model (model.name)
### MIND
- map dataset for MIND
- cache enabled