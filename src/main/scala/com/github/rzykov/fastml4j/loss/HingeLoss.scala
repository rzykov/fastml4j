package com.github.rzykov.fastml4j.loss

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
//
import com.github.rzykov.fastml4j.util.Implicits._

/**
  * Hinge Loss
  * http://www1.inf.tu-dresden.de/~ds24/lehre/ml_ws_2013/ml_11_hinge.pdf
  *
  * Created by rzykov on 31/05/17.
  */


class HingeLoss extends Loss {

  /**
    * Transform loss function from  {1, -1} to {0, 1}
    *
    * Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    * Therefore the gradient is -(2y - 1)*x
    *
    * @param labels of data
    * @return scaled labels
    */
  protected def labelScaled(labels: INDArray): INDArray = 2f * labels - 1f

  def pureLoss(weights: INDArray, dataSet: DataSet): INDArray = {
    val out = 1 + ((dataSet.getFeatureMatrix dot weights.T) * labelScaled(dataSet.getLabels)).neg()
    BooleanIndexing.replaceWhere(out, 0.0f, Conditions.lessThan(0.0f)) // condition 1-yt<0
    out.T
  } //max(0,1-y*yhat)

  override def loss(weights: INDArray, dataSet: DataSet): Float = {
    val scoreArr = pureLoss(weights, dataSet)
    scoreArr.sumFloat / dataSet.numExamples
  }

  override def gradient(weights: INDArray, dataSet: DataSet): INDArray = {
    val mask = pureLoss(weights, dataSet)
    BooleanIndexing.replaceWhere(mask, 1.0f, Conditions.greaterThan(0.0f)) //condition yt<1

    val main = - (dataSet.getFeatureMatrix muliColumnVector (labelScaled(dataSet.getLabels).T * mask).T).sum(0)
    main / dataSet.numExamples
  }
}

