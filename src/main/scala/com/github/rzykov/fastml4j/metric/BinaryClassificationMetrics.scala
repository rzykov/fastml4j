package com.github.rzykov.fastml4j.metric

import com.github.rzykov.fastml4j.classification.ClassificationModel
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * Calculates binary classification metrics based on a comparison of real and predicted labels
  *
  * @param realLabels NDArray of real labels
  * @param predictedLabels NDArray of predicted lables
  */

class BinaryClassificationMetrics(val realLabels: INDArray, val predictedLabels: INDArray) {

  /**
    * stores unique sorted values of predicted labels
    */
  lazy val binTreshholds: Seq[Float] = predictedLabels
      .ravel
      .toArray
      .flatten
      .distinct
      .sorted
      .reverse
      .toSeq

  /**
    * class contains the result of confusion matrix
    *
    * @param predictedClass predicted class value
    * @param realClass real class value
    * @param qty number of cases
    */

  case class Confusion(predictedClass: Float, realClass: Float, qty: Int)

  /**
    * Calculates confusion matrix for different levels of threshold
    *
    */

  lazy val confusionMatrixByThreshold: Seq[(Float, Seq[Confusion])] = {

    val zipped = Nd4j.hstack( predictedLabels, realLabels)
      .toArray

    binTreshholds.map { threshold =>
      val confusion = zipped
        .map { case Array(predicted, real) =>
          val predictedClass = if( predicted >= threshold ) 1 else 0
          ((predictedClass, real), 1) }
      .groupBy(_._1)
      .toSeq
      .map{ case ((predictedClass, real), values) => Confusion(predictedClass, real, values.map( _._2).sum)}

      (threshold, confusion) }
  }

  /**
    * Calculates Recall metrics for different levels of threshold
    * @return collection of floats with pair: (threshold, recall)
    */

  def recallByTreshold: Seq[(Float, Float)] = {

    val totalRealPositives: Int = confusionMatrixByThreshold.map (_._2).head
      .filter(_.realClass == 1)
      .map(_.qty)
      .sum

    confusionMatrixByThreshold.map { case (threshold, confusions) =>
        val truePositives = confusions
          .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
          .map(_.qty)
          .sum

      (threshold, if(truePositives == 0) 0 else truePositives.toFloat / totalRealPositives.toFloat)}
  }

  /**
    * Calculates Precision metrics for different levels of threshold
    * @return collection of floats with pair: (threshold, precision)
    */

  def precisionByThreshold: Seq[(Float, Float)] = {

    confusionMatrixByThreshold.map { case (threshold, confusions) =>
      val truePositives = confusions
        .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
        .map(_.qty)
        .sum

      val allPositives = confusions
        .filter { c => c.predictedClass == 1 }
        .map(_.qty)
        .sum

      (threshold, if(truePositives == 0) 0 else truePositives.toFloat / allPositives.toFloat)}
  }

  /**
    * Computes the ROC curve
    * @return collection of pairs (fpr, tpr)
    */

  def roc: Seq[(Float, Float)] = {

    val rocCurve = confusionMatrixByThreshold.map { case (threshold, confusions) =>
      val truePositives = confusions
        .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
        .map(_.qty)
        .sum

      val trueNegatives = confusions
        .filter { c => c.predictedClass == c.realClass && c.realClass == 0 }
        .map(_.qty)
        .sum

      val falsePositives = confusions
        .filter { c => c.predictedClass != c.realClass && c.predictedClass == 1 }
        .map(_.qty)
        .sum

      val falseNegatives = confusions
        .filter { c => c.predictedClass != c.realClass && c.predictedClass == 0 }
        .map(_.qty)
        .sum

      val tpr = if(truePositives == 0) 0.0f else truePositives.toFloat / (truePositives + falseNegatives)
      val fpr = if(falsePositives == 0) 0.0f else falsePositives.toFloat / (trueNegatives + falsePositives)


       (fpr, tpr)  }

    ((0.0f, 0.0f)) +: rocCurve :+ ((1.0f, 1.0f))
  }

  /**
    * calculates AUC value (area under ROC curve)
    * @return AUC
    */
  def aucRoc: Float =
    roc.sortBy( x => (x._1, x._2))
      .sliding(2)
      .map { case Seq((leftX, leftY), (rightX, rightY)) =>  (rightX - leftX) * (rightY + leftY ) / 2 }
      .reduce(_+_)


}


