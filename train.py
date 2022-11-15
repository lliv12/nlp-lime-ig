from models import BasicDANModel
from data_loader import ReviewsDataset, EssaysDataset
from captum.attr import IntegratedGradients


dataset = ReviewsDataset()
model = BasicDANModel(dataset.vocab_size())

ex = dataset[45]
print(ex[0])
print(model(ex[0]))
