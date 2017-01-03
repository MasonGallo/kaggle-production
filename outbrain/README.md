# [Outbrain Competition](https://www.kaggle.com/c/outbrain-click-prediction)

Difficulty: 

- over 100GB of training data
- data spread across many files, requiring lots of join operations
- lots of sparsity

# My simple idea:

We use an ad's regularized past CTR performance for our predictions. In order to prevent low
memory issues, we use the smallest data types possible and make sure to delete
data structures as soon as we're done with them.

`outbrain.py` will make the predictions and generate an appropriately formatted
submission.
