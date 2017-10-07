# Fastml4j
Fast Scala and nd4j based machine learning framework

Done:
* Logistic regression
* Linear regression (ordinary least squares)
* Linear SVM: Hinge Loss with Pegasos SGD
* Classification metrics
* Implicit expressions for nd4j
* Regression metrics
* Migrate to Float 
* Port Hinge loss from {1, -1} to {0, 1}
* Move L2 regularisation to trait
* Got rid of nd4s, implement own implicits directly 


TODO:

* Migrate to 0.9.1 or 0.9.2 of nd4j
* [Bug] Delete intercept from regularisation
* Add Normalisation https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/preprocessor/AbstractDataSetNormalizer.java
* Write good tests for linear regression
* Stohastic gradient descent (with replacement and not)
* Decision tree
* Write documentation in code
* Boosting: Ada Boost
* Bagging: Random forest
* Gradient boosting trees
* PCA
* L1  regularisation? 
