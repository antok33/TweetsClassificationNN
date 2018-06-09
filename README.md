# Text classifier for tweets:
## Text classifier implementation for tweets (positive, negative, neutral), using a neural network (MLP). 

* MLP structure:

![MLP Image](https://github.com/antok33/TweetsClassificationNN/blob/master/NN.png)

    * Hidden layer 1: 120 units
    * Hidden layer 2: 90 units
    * Hidden layer 3: 60 units
 
* Other Parameters
    * activation function: relu
    * solver: adam [Paper: ADAM A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf)
    
* For the dataset click [here](http://alt.qcri.org/semeval2016/task4/).

* Finally, we compare the MLP with two simple Baseline Agorithms using as evaluation metrics: precision, recall, f1 score.
