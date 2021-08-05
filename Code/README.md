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
- [ ] auto-regressive user modeling
- [ ] distributed training
- [x] **config.filter_num -> config.hidden_dim**
- [x] **config.signal_length**
- [x] **config.embedding_dim**
- [ ] modify load_config
- [ ] cache dataset

## Issue
- the gradient after docReducer is sharp
- if we concat title with abstract, then there must be many duplicated words, how about removing them when preprocessing?
- bert tokenize in stream or in bunch?
- **connect to ps0 or cuda0?**
- **how to connect? use what shell**
- **your .ssh mode**
去掉-i
- **where to save codes and data?**
data job

in SpeedyFeed
- **many unused scripts and modules**
- **should we partition the dataloader?**
- **should we partition batch_size?**
- **is torch.cuda.set_device necessary?**
- data for speedyfeed
- can I run on cpu?

## Need to update
- [ ] Encoders.MHA, NPA, Pipeline
