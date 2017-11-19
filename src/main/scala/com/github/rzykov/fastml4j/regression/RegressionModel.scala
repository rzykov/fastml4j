package com.github.rzykov.fastml4j.regression

import com.github.rzykov.fastml4j.util.Intercept
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

abstract class RegressionModel  {
  /** contains weights after the fitting the model*/
  var weights: INDArray = Nd4j.zeros(1)

  /** contains changes of the loss function value during optimisation at every step*/
  var losses: Seq[Float] = Seq[Float]()

  /** fit weights to the data provided by the implementation of the model
    * @param dataSet input DataSet with features and labels
    * @param initWeights initial weights, None if omitted
    * */
  def fit(dataSet: DataSet, initWeights: Option[INDArray]): Unit

  /** predict output for a vector of data
    * @param inputVector INDarray data vector
    *
    * @return output of the model
    * */
  def predict(inputVector: INDArray): Float

  /** predict the probability of positive class for one vector of data
    * @param dataSet dataset with features and labels
    *
    * @return NDArray with output of the model
    * */
  def predict(dataSet: DataSet): INDArray

}
