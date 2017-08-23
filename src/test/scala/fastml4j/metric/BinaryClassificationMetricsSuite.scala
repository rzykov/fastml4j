package fastml4j.metric

import org.nd4s.Implicits._
import org.scalatest.Matchers._
import org.scalatest._

import math.random
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

class BinaryClassificationMetricsSuite extends FunSuite {

  def assertSeq(a: Seq[Double], b: Seq[Double]): Unit = assert(a.zip(b).forall{ case(aX, bX) => aX === bX +- 0.01} )
  def assertTupleSeq(a: Seq[(Double, Double)], b: Seq[(Double, Double)]): Unit =
    assert(a.zip(b).forall{ case((aX, aY), (bX, bY)) => aX === bX +- 0.01 && aY === bY +- 0.01 } )


  private def validateMetrics(metrics: BinaryClassificationMetrics,
    expectedThresholds: Seq[Double],
    expectedROCCurve: Seq[(Double, Double)],
    expectedPrecisions: Seq[Double],
    expectedRecalls: Seq[Double]): Unit = {

    assertSeq(metrics.binTreshholds, expectedThresholds)
    assertTupleSeq(metrics.precisionByTreshhold, expectedThresholds.zip(expectedPrecisions))
    println(metrics.recallByTreshhold)
    println(expectedRecalls)

    assertTupleSeq(metrics.recallByTreshhold, expectedThresholds.zip(expectedRecalls))
    assertTupleSeq(metrics.roc, expectedROCCurve)
  }


  // Tests have been taken from Spark:
  // https://github.com/apache/spark/blob/c64a8ff39794d60c596c0d34130019c09c9c8012/mllib/src/test/scala/org/apache/spark/mllib/evaluation/BinaryClassificationMetricsSuite.scala

  test("binary evaluation metrics") {
    val predictions = Array(0.1, 0.1, 0.4, 0.6, 0.6, 0.6, 0.8).map(Array(_)).toNDArray
    val real = Array(0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0).map(Array(_)).toNDArray
    val metrics = new BinaryClassificationMetrics(real, predictions)

    val thresholds = Seq(0.8, 0.6, 0.4, 0.1)
    val numTruePositives = Seq(1, 3, 3, 4)
    val numFalsePositives = Seq(0, 1, 2, 3)
    val numPositives = 4
    val numNegatives = 3
    val precisions = numTruePositives.zip(numFalsePositives).map { case (t, f) => t.toDouble / (t + f)}
    val recalls = numTruePositives.map(t => t.toDouble / numPositives)
    val fpr = numFalsePositives.map(f => f.toDouble / numNegatives)
    val rocCurve = Seq((0.0, 0.0)) ++ fpr.zip(recalls) ++ Seq((1.0, 1.0))

    validateMetrics(metrics, thresholds, rocCurve, precisions, recalls)
  }


  test("binary evaluation metrics  where all examples have positive label") {
    val predictions = Array(0.5, 0.5).map(Array(_)).toNDArray
    val real = Array(1.0, 1.0).map(Array(_)).toNDArray
    val metrics = new BinaryClassificationMetrics(real, predictions)

    val thresholds = Seq(0.5)
    val precisions = Seq(1.0)
    val recalls = Seq(1.0)
    val fpr = Seq(0.0)
    val rocCurve = Seq((0.0, 0.0)) ++ fpr.zip(recalls) ++ Seq((1.0, 1.0))

    validateMetrics(metrics, thresholds, rocCurve, precisions, recalls)
  }

  test("binary evaluation metrics  where all examples have negative label") {
    val predictions = Array(0.5, 0.5).map(Array(_)).toNDArray
    val real = Array(0.0, 0.0).map(Array(_)).toNDArray
    val metrics = new BinaryClassificationMetrics(real, predictions)

    val thresholds = Seq(0.5)
    val precisions = Seq(0.0)
    val recalls = Seq(0.0)
    val fpr = Seq(1.0)
    val rocCurve = Seq((0.0, 0.0)) ++ fpr.zip(recalls) ++ Seq((1.0, 1.0))

    validateMetrics(metrics, thresholds, rocCurve, precisions, recalls)
  }

  test("aucRoc random") {

    val real = Array(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0).map(Array(_)).toNDArray

    val predictionsRandom = Array( 0.5, 0.5 , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5).map(Array(_)).toNDArray
    val metricsRandom = new BinaryClassificationMetrics(real, predictionsRandom)
    assert(metricsRandom.aucRoc === 0.5 +- 0.05)

    val predictionsNegative = Array(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0).map(Array(_)).toNDArray
    val metricsNegative = new BinaryClassificationMetrics(real, predictionsNegative)
    assert(metricsNegative.aucRoc === 0.0 +- 0.05)

    val predictionsTrue = Array(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0).map(Array(_)).toNDArray
    val metricsTrue = new BinaryClassificationMetrics(real, predictionsTrue)
    assert(metricsTrue.aucRoc === 1.0 +- 0.05)


  }

}
