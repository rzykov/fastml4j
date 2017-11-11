package com.github.rzykov.fastml4j.metric

import com.github.rzykov.fastml4j.classification.ClassificationModel
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * Calculates regression metrics based on a comparison of real and predicted labels
  *
  * @param realLabels NDArray of real labels
  * @param predictedLabels NDArray of predicted lables
  */

class RegressionMetrics(val realLabels: INDArray, val predictedLabels: INDArray) {

  private lazy val numCases: Float = realLabels.rows().toFloat
  require(numCases > 1, "use matrix rather than vectors")

  /**
    * calculates root of sum of squared differences
    */

  lazy val rootDifferenceSumSquared: Float =  Nd4j.norm2(realLabels - predictedLabels).sumFloat

  /**
    * calculates sum of squared differences
    */

  lazy val differenceSumSquared: Float =  rootDifferenceSumSquared * rootDifferenceSumSquared

  /**
    * calculates the variance of errors (differences)
    */

  lazy val differenceVariance: Float = (realLabels - predictedLabels).variance
  /**
    * Calculates MSE
    * @return MSE
    */

  def meanSquaredError: Float = differenceSumSquared / numCases
  /**
    * Calculates RMSE
    * @return RMSE
    */
  def rootMeanSquaredError: Float = rootDifferenceSumSquared / math.sqrt(numCases).toFloat

  /**
    * Calculates Mean Absolute Error(MAE)
    * @return MAE
    */

  lazy val meanAbsoluteError: Float = Nd4j.norm1(realLabels - predictedLabels).sumFloat

  override def toString = s"rootMeanSquaredError: $rootMeanSquaredError  \nmeanSquaredError $meanSquaredError \nmeanAbsoluteerror $meanAbsoluteError\n"

}
