package fastml4j.losses

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions



/**
  * Created by rzykov on 31/05/17.
  */


class OLSLoss(lambdaL2: Double) extends Loss {

  def loss(weights: INDArray, trainData: INDArray, labels: INDArray): Double = {
    val predictedVsActual = (weights dot trainData.T) - labels
    val regularized: Double =  (weights * weights).sumT[Double] * lambdaL2 / 2

    (predictedVsActual.T * predictedVsActual).sumT[Double] / 2.0 / (trainData.rows)  + regularized
  }

  def gradient(weights: INDArray, trainData: INDArray, labels: INDArray): INDArray = {
    val main = trainData.T dot ((trainData dot weights.T) - labels)
    val regularized = weights * lambdaL2

    (main / (trainData.rows)).T + regularized
  }

}

