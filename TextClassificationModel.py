from torch import nn

#classification model in this file based on
#https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2(03.01.2023)
#and https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html (03.01.2023)

# Create model
device = "cpu"


# Define model
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # print("Text: ", text)
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
