# Shoes Reviews class prediction with differnt Natural Language Processing models.


In this project I build two different models to predict the sentiment of a running shoe review.
I clustered the running reviews in 3 classes: ***Positive, Neutral*** and ***Negative***.

 1.[`GloVe` embeddings + LSTM model](https://nlp.stanford.edu/projects/glove/)
 
 GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

 2.[TensorflowHub `nnlm-en-dim128` embeddings + FFD neural network with dropout](https://tfhub.dev/google/nnlm-en-dim128/2)

 nnlm-en-dim128 is a token based text embedding trained on English Google News 200B corpus.

 ## Results and Metrics
 I could obtain 0.7 accuracy for GloVe and LSTM, and 0.78 for TensorflowHub embeddings and FFD neural networks on the validation samples.
