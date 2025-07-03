

## Machine Learning Research

This project requires us to make predictions based on a variety of factors in a football game to predict the outcome of an offensive play (pass/run) as well as specific play types. Given the multiple factors, what kind of model or strategy should be use to predict these outcomes? This research should help us answer those questions.

## Neural Networks

- A neural network is a machine learning model that mimics how the human brain stores and infers information. It is made up of an input layer for your data, multiple "hidden" layers for processing, and an output layer for the result.
- "Each node connects to others, and has its own associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network." via: https://www.ibm.com/topics/neural-networks
- Neural networks are trained on training data which helps them learn and improve their accuracy.
- Neural networks work very well with large datasets and when computational power is not an issue.
- Neural networks may struggle when there is a small amount of data, or if the data may be prone to overfitting (ie noise).


A neural network may be preferable for our data at least to start, given we want to begin by predicting a binary outcome in pass/run. We may have some concerns about small data size, however we could potentially use generative AI to create synthetic data.

## Decision Trees & Random Forests

- A decision tree is a machine learning model that follows a flowchart-like structure. It consists of a **root** node, which represents the dataset at whole, **internal** nodes that represent decisions or tests on each attribute, **branches**, which represent the outcome of each test, and **leaf** nodes, which represent the final decision of the test.
- How Decision Trees Work? The process of creating a decision tree involves:Selecting the Best Attribute: Using a metric like Gini impurity, entropy, or information gain, the best attribute to split the data is selected.Splitting the Dataset: The dataset is split into subsets based on the selected attribute.
Repeating the Process: The process is repeated recursively for each subset, creating a new internal node or leaf node until a stopping criterion is met (e.g., all instances in a node belong to the same class or a predefined depth is reached). via: https://www.geeksforgeeks.org/decision-tree/
- A random forest is a machine learning model that implements many decision trees to make predictions. 
- Decision Trees & Random Forests may work better than Neural Networks when we are working with small data or may be less prone to overfitting compared to neural networks.
- Decision Trees & Random Forests may require more tuning of hyperparameters as well as greater complexity for decisions.

## Implementation

- Libraries like NumPy can offer pre-created models that we can plug our data into and make those predictions. This will allow us to spend less time in development, and more time focusing on cleaning the data and fine tuning to try and increase prediction accuracy. However, ML libraries are not very customizable, which may pose an issue for our very specific dataset. These libraries also typically obscure what is actually happening to the data behind-the-scenes, given that this is a computer science project that may be a bit of an underwhelming approach for a capstone project.
- Using a custom solution in another language like C++ offers us full control of our model, and gives us experience actually working with the algorithms behind how we are making our predictions. This may prove more beneficial in the long run, as we can tailor everything to our specific data, however this may come with much greater time in development to reach that greater level of accuracy.

## Conclusions

I am personally leaning towards using a neural network, at least for our initial pass/run prediction. We can change this approach later if we find it is not working out. As for implementation, I believe we may be better off implementing this ourselves. There are lots of samples online to look to, (https://github.com/Cr4ckC4t/neural-network-from-scratch, **post Dr. Rao Assignment Here**) and this will give us more control over what we are doing with our data, which will allow us to learn more about what goes on in depth in these models. Open to other thoughts.