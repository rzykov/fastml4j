package fastml4j.losses

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions



/**
  * Created by rzykov on 31/05/17.
  */


class OLSLoss(lambdaL2: Double) extends Loss {

  def loss(weights: INDArray, dataSet: DataSet): Double = {
    val predictedVsActual = (weights dot dataSet.getFeatures.T) - dataSet.getLabels.T
    val regularized: Double =  (weights * weights).sumT[Double] * lambdaL2 / 2

    (predictedVsActual.T * predictedVsActual).sumT[Double] / 2.0 / (dataSet.numExamples)  + regularized
  }

  def gradient(weights: INDArray, dataSet: DataSet): INDArray = {
    val main = dataSet.getFeatures.T dot ((dataSet.getFeatures dot weights.T) - dataSet.getLabels.T)
    val regularized = weights * lambdaL2

    (main / (dataSet.getFeatures.rows)).T + regularized
  }

}

