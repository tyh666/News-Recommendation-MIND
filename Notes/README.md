# Blogs
- [VAE](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
  - [Variational Inference](https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29)
- [Document Summarization](https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25)
- [Makov Chain](https://towardsdatascience.com/brief-introduction-to-markov-chains-2c8cab9c98ab)
- [Capsule Net](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)

# Insights
- multi-task, one for document compression, another for recommendation
  - pipeline or end-to-end
- encoder-decoder break apart, the encoder consumes the input and forms a hidden representation, the decoder is another encoder to take in the standard output and generates another hidden representation. The objective is to minimize the difference between the two hidden representation. Bayesian is used for inference.