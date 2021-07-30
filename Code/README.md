## Preprocess
- [ ] remove stop words
- [ ] add special [END] token

## News Encoder
- [x] CNN encoder
- [ ] FIM encoder
- [ ] Bert encoder
  - [ ] [CLS] as news repr
  - [ ] attention over the last layer as news repr

## User Encoder
- [x] RNN encoder
- [ ] LSTUR
- [ ] Adaptive and time-aware LSTUR, see [29]
- [ ] NRMS

## Document Reducer
- [ ] matching based
  - [ ] only topk
  - [ ] topk with threshold
  - [ ] diversified
    - [ ] use document vector as an extraction input
    - [ ] design methods to extract non-duplicated terms within an article
  - [ ] dynamic allocation of top k
- [ ] Seq2Seq based
- [ ] RL based

## Term Fuser
- [ ] differentiable and efficient

## Interactor
- [ ] Bert
  - [ ] position embedding within each news
  - [ ] order embedding across the whole user history
- [ ] FIM
- [ ] KNRM

## Workflow
- [ ] extract terms from every historical news when the history is updated
- [ ] extract terms incrementally
- [ ] long and short term extraction
- [ ] auto-regressive user modeling


## Issue
- the gradient after docReducer is sharp
- bert tokenizer and regular tokenizer are different
- input words is wrong, we can only input embeddings