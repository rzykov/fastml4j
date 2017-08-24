package fastml4j.metric

import fastml4j.classification.ClassificationModel
import fastml4j.implicits.RichIndarray._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._


class BinaryClassificationMetrics(val outcome: INDArray, val predictedLabels: INDArray) {

  lazy val binTreshholds: Seq[Double] = predictedLabels
      .ravel
      .toArray
      .flatten
      .distinct
      .sorted
      .reverse
      .toSeq


  case class Confusion(predictedClass: Double, realClass: Double, qty: Int)

  lazy val confusionMatrixByThreshold: Seq[(Double, Seq[Confusion])] = {

    val zipped = Nd4j.hstack( predictedLabels, outcome)
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


  def recallByTreshhold: Seq[(Double, Double)] = {

    val totalRealPositives: Int = confusionMatrixByThreshold.map (_._2).head
      .filter(_.realClass == 1)
      .map(_.qty)
      .sum

    confusionMatrixByThreshold.map { case (threshold, confusions) =>
        val truePositives = confusions
          .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
          .map(_.qty)
          .sum

      (threshold, if(truePositives == 0) 0 else truePositives.toDouble / totalRealPositives.toDouble)}
  }

  def precisionByTreshhold: Seq[(Double, Double)] = {

    confusionMatrixByThreshold.map { case (threshold, confusions) =>
      val truePositives = confusions
        .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
        .map(_.qty)
        .sum

      val allPositives = confusions
        .filter { c => c.predictedClass == 1 }
        .map(_.qty)
        .sum

      (threshold, if(truePositives == 0) 0 else truePositives.toDouble / allPositives.toDouble)}
  }

  def roc: Seq[(Double, Double)] = {

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

      val tpr = if(truePositives == 0) 0.0 else truePositives.toDouble / (truePositives + falseNegatives)
      val fpr = if(falsePositives == 0) 0.0 else falsePositives.toDouble / (trueNegatives + falsePositives)


       (fpr, tpr)  }

    (0.0, 0.0) +: rocCurve :+ (1.0, 1.0)
  }

  def aucRoc: Double =
    roc.sortBy( x => (x._1, x._2))
      .sliding(2)
      .map { case Seq((leftX, leftY), (rightX, rightY)) =>  (rightX - leftX) * (rightY + leftY ) / 2 }
      .reduce(_+_)


}


