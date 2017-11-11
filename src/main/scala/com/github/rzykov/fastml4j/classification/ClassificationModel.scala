package com.github.rzykov.fastml4j.classification

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import com.github.rzykov.fastml4j.util.Implicits._
import com.github.rzykov.fastml4j.util.Intercept

/**
  * Abstract class for any classification model
  */
abstract class ClassificationModel{
  /** contains weights after the fitting the model*/
  var weights: INDArray = Nd4j.zeros(1)

  /** contains changes of the loss function value during optimisation at every step*/
  var losses: Seq[Float] = Seq[Float]()

  /** fit weights to the data provided by the implementation of the model
    * @param dataSet input DataSet with features and labels
    * @param initWeights initial weights, None if omitted
    * */
  def fit(dataSet: DataSet, initWeights: Option[INDArray]): Unit

  /** predict class (1 or 0) for one vector of data
    * @param inputVector INDarray data vector
    *
    * @return predicted class
    * */

  def predictClass(inputVector: INDArray): Float

  /** predict the probability of positive class for one vector of data
    * @param inputVector INDarray data vector
    *
    * @return probability of positive class
    * */

  def predict(inputVector: INDArray): Float

}

