package mlscala.losses

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions

/**
  * Created by rzykov on 31/05/17.
  */


class HingeLoss(lambdaL2: Double) extends Loss {

  def pureLoss(weights: INDArray, data: INDArray, labels: INDArray): INDArray = {
    val out = ((data dot weights.T) * labels).neg() + 1
    BooleanIndexing.replaceWhere(out, 0.0, Conditions.lessThan(0.0)) // condition 1-yt<0
    out.T
  } //max(0,1-y*yhat)


  def loss(weights: INDArray, trainData: INDArray, labels: INDArray): Double = {
    val scoreArr = pureLoss(weights, trainData, labels)
    val main: Double = scoreArr.sumT[Double]
    val regularized: Double =   (weights * weights).sumT[Double] * lambdaL2 / 2

    (main / trainData.rows) + regularized
  }


  def gradient(weights: INDArray, trainData: INDArray, labels: INDArray): INDArray = {
    val mask = pureLoss(weights, trainData, labels)
    BooleanIndexing.replaceWhere(mask, 1.0, Conditions.greaterThan(0.0)) //condition yt<1

    val main = (trainData muliColumnVector (labels * mask).T * (-1.0)).sum(0)

    //val main = (labels * mask).neg()
    val regularized = weights * lambdaL2

    (main / trainData.rows + regularized)
  }

}

