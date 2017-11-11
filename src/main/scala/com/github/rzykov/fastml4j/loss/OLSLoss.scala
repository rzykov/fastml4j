package com.github.rzykov.fastml4j.loss

//
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import com.github.rzykov.fastml4j.util.Implicits._


/**
  * Created by rzykov on 31/05/17.
  */


class OLSLoss extends Loss {

  override def loss(weights: INDArray, dataSet: DataSet): Float = {
    val predictedVsActual = (weights dot dataSet.getFeatures.T) - dataSet.getLabels.T

    (predictedVsActual * predictedVsActual).sumFloat / 2.0f / (dataSet.numExamples)
  }

  override def gradient(weights: INDArray, dataSet: DataSet): INDArray = {
    val main =((dataSet.getFeatures dot weights.T) - dataSet.getLabels.T) dot  dataSet.getFeatures
    (main / (dataSet.getFeatures.rows))
  }

}

