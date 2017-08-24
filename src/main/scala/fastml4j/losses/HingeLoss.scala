package fastml4j.losses


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4s.Implicits._
import fastml4j.implicits.RichIndarray._

/**
  * Created by rzykov on 31/05/17.
  */


class HingeLoss(lambdaL2: Double) extends Loss {

  //http://www1.inf.tu-dresden.de/~ds24/lehre/ml_ws_2013/ml_11_hinge.pdf

  def pureLoss(weights: INDArray, dataSet: DataSet): INDArray = {
    val out = 1 + ((dataSet.getFeatureMatrix dot weights.T) * dataSet.getLabels.T).neg()
    BooleanIndexing.replaceWhere(out, 0.0, Conditions.lessThan(0.0)) // condition 1-yt<0
    out.T
  } //max(0,1-y*yhat)


  def loss(weights: INDArray, dataSet: DataSet): Double = {
    val scoreArr = pureLoss(weights, dataSet)
    val main: Double = scoreArr.sumT[Double]
    val regularized: Double =   (weights * weights).sumT[Double] * lambdaL2 / 2

    (main / dataSet.numExamples) + regularized
  }


  def gradient(weights: INDArray, dataSet: DataSet): INDArray = {
    val mask = pureLoss(weights, dataSet)
    BooleanIndexing.replaceWhere(mask, 1.0, Conditions.greaterThan(0.0)) //condition yt<1

    val main = - (dataSet.getFeatureMatrix muliColumnVector (dataSet.getLabels.T * mask).T).sum(0)

    //val main = (dataSet.getLabels * mask).neg()
    val regularized = weights * lambdaL2

    (main / dataSet.numExamples + regularized)
  }

}

