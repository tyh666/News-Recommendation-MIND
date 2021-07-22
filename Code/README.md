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
  - [ ] dynamic allocation of top k
- [ ] Seq2Seq based
- [ ] RL based

## Term Fuser
- [ ] differentiable and efficient

## Interactor
- [ ] Bert
- [ ] FIM
- [ ] KNRM

## Workflow
- [ ] extract terms from every historical news when the history is updated
- [ ] extract terms incrementally
- [ ] long and short term extraction


## Issue
- the gradient after docReducer is sharp