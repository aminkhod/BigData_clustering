# K-means implementation in Spark - Python

This project was developed during my masters at Paris-Dauphine university. You can find [here](https://github.com/bilal-elchami/data-analytics-kmeans/blob/master/docs/report.pdf) the project report.

## Objectives
* Implement KMeans in Spark.
* Generate random data to feed our algorithms.
* Test the algorithms and show some resuts.

## K-means:
The K-means algorithm is one of the most used clustering methods to group a disparate set of data. This method relies mainly on unsupervised learning. The goal of this project is to implement the KMean algorithm with Spark (python) and evaluating the performance of our implementation based on existing and generated data.

## K-means++:
k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm. It was proposed in 2007 by David Arthur and Sergei Vassilvitskii, to avoid the sometimes poor clusterings found by the standard k-means algorithm.
You can find below a good example to show the weak initial values of the centroids.

<p align="center">
 <img src="https://raw.githubusercontent.com/bilal-elchami/data-analytics-kmeans/master/docs/img/initCentroids.PNG">
</p>

The "x"s represent the initial centers chosen randomly by the K-means algorithm. Although the algorithm converges and finds the horizontal groups, it is obvious that the vertical groups represent more accurate result.
This unreliable solution is due to the weak choice of the initial centers.

## Data generator:

In order to evaluate the implemented algorithms, a data generator has been set up.
We coded the 2 python scripts (`generator.py` and `generator-noise.py`) to create clean data files.
`generator.py` allows to generate "grouped points" and save the result in a csv file. Then the `generator-noise.py` scripts comes to generate some random noise around the grouped points.

Here are the steps of the data generation:

### Step 1 - Create random grouped data:

<p align="center">
 <img height="400" src="https://raw.githubusercontent.com/bilal-elchami/data-analytics-kmeans/master/docs/img/generator-points.png">
</p>

In this case we are generating 9 points clustered in 3 groups with a standard deviation equals to 2 saved in out.csv file by executing the following command:

```
$ spark-submit generator.py out 9 3 2 10
```

### Step 2 - Generate noise around the points:

<p align="center">
 <img height="400" src="https://raw.githubusercontent.com/bilal-elchami/data-analytics-kmeans/master/docs/img/generator-noises.png">
</p>

In this case we are generating 9 points representing the noise in 3 groups with a standard deviation equals to 2 saved in out.csv file by executing the following command:

```
$ spark-submit generator_noise.py out 9 3 2 10
```

## Experiments

Here are the iterations of solving the iris-sepal dataset:

<p align="center">
 <img height="400" src="https://raw.githubusercontent.com/bilal-elchami/data-analytics-kmeans/master/iterations-results/sepal/iris-sepal-clustering.gif">
</p>

There is more experiment results in the `iterations-results` directory
