# This branch is MY WORK
Since the paper is under review, I cannot release more details.

## Environment
```
python=3.8.11
torch==1.9.1
```

## Instruction
```bash
cd Code

# train
python tesrec.py

# test
python tesrec.py -m test
```

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