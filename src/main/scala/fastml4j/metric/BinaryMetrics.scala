package fastml4j.metric

import fastml4j.classification.ClassificationModel
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.RichIndarray._

//Classification:
// Recall
// F1
// Missclassifiaction matrix
// Precision
// AUC
// ROC table


// Regression
// Squared loss
// R2 share

// Ranking
// NDCG
// MAR??



class BinaryMetrics(val outcome: INDArray, val predictedLabels: INDArray, val bins: Int) {

  //def this(dataSet: DataSet, predictor: ClassificationModel) = this(dataSet, predictor.predict(dataSet))
  require(bins > 0, "bins must be > 0")

  private lazy val binTreshholds: Seq[Double] = {
   val outputs = outcome
      .ravel
      .toArray
      .flatten
      .distinct
      .sorted

    require( outputs.size >= bins, s"Output distinct values ${outputs.size} less than bins $bins"  )

    val interval = outputs.size.toFloat / bins.toFloat
    outputs.zipWithIndex
      .map { case(value, index) => (value, (index / interval).toInt) }
      .groupBy (_._2)
      .map { case(bin, values) => values.map(_._1).max }
      .toSeq
  }

  case class Confusion(predictedClass: Double, realClass: Double, qty: Int)

  lazy val confusionMatrixByThreshold: Seq[(Double, Seq[Confusion])] = {
    val zipped = Nd4j.hstack(outcome, predictedLabels)
      .toArray

    binTreshholds.map { threshold =>
      val confusion = zipped.map { case Array(predicted, real) =>
        val predictedClass = if( predicted < threshold ) 0 else 1
        ((predictedClass, real), 1) }
      .groupBy(_._1)
      .toSeq
      .map{ case ((predictedClass, real), values) => Confusion(predictedClass, real, values.map( _._2).sum)}

      (threshold, confusion) }
  }


  val recallByTreshhold: Seq[(Double, Double)] = {

    lazy val totalRealPositives: Int = confusionMatrixByThreshold.map (_._2).head
      .filter(_.realClass == 1)
      .map(_.qty)
      .sum

    confusionMatrixByThreshold.map { case (threshold, confusions) =>
        val truePositives = confusions
          .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
          .map(_.qty)
          .sum
      (threshold, truePositives.toDouble / totalRealPositives.toDouble)}
  }

  val precisionByTreshhold: Seq[(Double, Double)] = {

    confusionMatrixByThreshold.map { case (threshold, confusions) =>
      val truePositives = confusions
        .filter { c => c.predictedClass == c.realClass && c.realClass == 1 }
        .map(_.qty)
        .sum

      val allPositives = confusions
        .filter { c => c.predictedClass == 1 }
        .map(_.qty)
        .sum

      (threshold, truePositives.toDouble / allPositives.toDouble)}
  }



}


