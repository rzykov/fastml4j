package mlscala.losses

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * Created by rzykov on 31/05/17.
  */


class LogisticLoss(lambdaL2: Double) extends Loss {

  def sigmoid(weights: INDArray, data: INDArray): INDArray = Transforms.sigmoid(data dot weights.T)


  def loss(weights: INDArray, trainData: INDArray, labels: INDArray): Double = {
    val sigmoidVec = sigmoid(weights, trainData)
    val lossVec = (labels * Transforms.log(sigmoidVec)) - (labels - 1.0) * Transforms.log((sigmoidVec - 1.0).neg())
    val regularized: Double = (weights * weights).sumT[Double] * lambdaL2 / 2

     - (lossVec.sumT[Double] / trainData.rows) + regularized
  }


  def gradient(weights: INDArray, trainData: INDArray, labels: INDArray): INDArray = {
    val regularized = weights * lambdaL2

    ///( trainData.rows + regularized)
    weights
  }

}

