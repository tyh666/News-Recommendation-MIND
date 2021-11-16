## Environment
``` bash
git config --global user.name 'namespace-Pt'
git config --global user.email 'namespace.pt@gmail.com'
sudo apt-get install screen -y
sudo apt-get install rsync -y
conda init
echo 'alias nn="conda activate /data/v-pezhang/nn"' >> ~/.bashrc
echo 'alias pt="screen -r -d pt"' >> ~/.bashrc
echo 'alias gf="conda activate /data/v-pezhang/gf"' >> ~/.bashrc

source ~/.bashrc
sleep infinity
```

## Instruction
```bash
cd /data/v-pezhang/Code/Document-Reduction/Code
python tesrec.py -ws=2
python plm.py -ws=2 -st=0 -bs=16
python xformer.py -ws=2 -sl=40 -b=bigbird

python tesrec.py -s demo -bs=5 -is=10

python tesrec.py -m encode
python tesrec.py -m analyse -ck=230000

python tesrec.py -it=dev -ck=230000
python tesrec.py -it=test -ck=230000 -n=38

python tesrec.py -rt=sd -ck=230000

python tesrec.py -m test -ck=589
python plm.py -m test -ck=589
python esm -m test -ck=150000
```

## Insights of GateFormer
- the selected terms largely depends on the data, unlike bm25 that panalize widely appeared terms, our method learn to balance the weight of each term
- the selected terms of one piece of news inclines to be the same among different users

## TODO
- [ ] whether to mention cascaded model in IR, threshold gating in recommendation
- [ ] why bigbird slow -> it repetitively computes linear complexity attention

## Preprocess
- tokenize all news articles, bm25 reordered articles, entities and pre-defined keywords
- generate cache

## Embedding
- bert word embedding

## News Encoder
- CNN
- RNN
- Multi Head Attention
- One layer bert
- Pooling

## User Encoder
- LSTM
- MHA
- LSTUR
- Pooling

## Document Reducer
- [ ] extract terms incrementally
- [ ] long and short term extraction
- [ ] Seq2Seq based
- [ ] RL based

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