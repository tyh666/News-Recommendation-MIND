# Blogs
- [VAE](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
  - [Variational Inference](https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29)
- [Document Summarization](https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25)
- [Makov Chain](https://towardsdatascience.com/brief-introduction-to-markov-chains-2c8cab9c98ab)
- [Capsule Net](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)
- [Cross Entropy](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- [Policy Gradient](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c)
- [lda2vec](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)

## Insights
- multi-task, one for document compression, another for recommendation
  - pipeline or end-to-end
- encoder-decoder break apart, the encoder consumes the input and forms a hidden representation, the decoder is another encoder to take in the standard output and generates another hidden representation. The objective is to minimize the difference between the two hidden representation. Bayesian is used for inference.
- multiple vectors for one term, a position-aware one, a semantic-aware one, a syntax aware one
- Sequential recommendation
- use knowledge graph to model hiearchical interest
- @dou
  - 提取关键词, 在工业界是普遍做法, 创新性不够
  - 还是要利用整篇文章的信息
  - retrieval阶段可以研究研究

- use pointer network to make the decoding differentiable
- multi-layer pointer network (transformer)
## Question
- we compress the news before interaction, or within the interaction
- what is the difference between index-selection and REINFORCE sampling?