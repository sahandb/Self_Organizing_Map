# Self_Organizing_Map
I applied a Self-Organizing Map (SOM net) to a dataset.


I use a dataset which contains 1000 samples from 4 classes that are not linearly separable.
Split the data into training and test sets. 
Select 20% of samples for test data. Build a SOM with a 10x10 grid topology to learn the 2D distribution of the data.
Use the Euclidean distance to determine the winner neuron in the output (map) layer.
I reduce the size of the neighborhood and the learning rate in consecutive epochs.

At first I split the data into training and test sets. choose 20% of samples for test data then build a SOM with a 10x10 grid topology to learn the 2D distribution of the data. Use the Euclidean distance to determine the winner neuron in the output (map) layer and use 3 size of neighbors like 1 2 5 for experiment the difference size of the neighborhood and learning rate was 0.5 for all and train model for 5000 iteration. 

Weights every time initial randomly along -1 and 1 in x and y axis

At the end determine the winner neuron for every sample and sum all of the distance value for them

I use pre defined library for SOM in python (miniSOM)

![image](https://user-images.githubusercontent.com/24508376/219628446-7cd88ab3-f938-4757-bd53-af86de95bd79.png)


![image](https://user-images.githubusercontent.com/24508376/219628634-5a6a7aa8-dfd8-4f10-86c8-21b508bf9da9.png)


![image](https://user-images.githubusercontent.com/24508376/219628730-30f2f4c0-11d7-4ad8-accb-e996a109cec2.png)

