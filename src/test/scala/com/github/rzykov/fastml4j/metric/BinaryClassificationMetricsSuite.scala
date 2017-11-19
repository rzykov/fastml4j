package com.github.rzykov.fastml4j.metric

import com.github.rzykov.fastml4j.metric.BinaryClassificationMetrics
import org.scalatest.Matchers._
import org.scalatest._

import math.random
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

class BinaryClassificationMetricsSuite extends FunSuite {

  def assertSeq(a: Seq[Float], b: Seq[Float]): Unit = assert(a.zip(b).forall{ case(aX, bX) => aX === bX +- 0.01f} )
  def assertTupleSeq(a: Seq[(Float, Float)], b: Seq[(Float, Float)]): Unit =
    assert(a.zip(b).forall{ case((aX, aY), (bX, bY)) => aX === bX +- 0.01f && aY === bY +- 0.01f } )


  private def validateMetrics(metrics: BinaryClassificationMetrics,
    expectedThresholds: Seq[Float],
    expectedROCCurve: Seq[(Float, Float)],
    expectedPrecisions: Seq[Float],
    expectedRecalls: Seq[Float]): Unit = {

    assertSeq(metrics.binThresholds, expectedThresholds)
    assertTupleSeq(metrics.precisionByThreshold, expectedThresholds.zip(expectedPrecisions))

    assertTupleSeq(metrics.recallByThreshold, expectedThresholds.zip(expectedRecalls))
    assertTupleSeq(metrics.roc, expectedROCCurve)
  }


  // Tests have been taken from Spark:
  // https://github.com/apache/spark/blob/c64a8ff39794d60c596c0d34130019c09c9c8012/mllib/src/test/scala/org/apache/spark/mllib/evaluation/BinaryClassificationMetricsSuite.scala

  test("binary evaluation metrics") {
    val predictions = Array(0.1, 0.1, 0.4, 0.6, 0.6, 0.6, 0.8).map(Array(_)).toNDArray
    val real = Array(0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0).map(Array(_)).toNDArray
    val metrics = new BinaryClassificationMetrics(real, predictions)

    val thresholds = Seq[Float](0.8f, 0.6f, 0.4f, 0.1f)
    val numTruePositives = Seq(1, 3, 3, 4)
    val numFalsePositives = Seq(0, 1, 2, 3)
    val numPositives = 4
    val numNegatives = 3
    val precisions = numTruePositives.zip(numFalsePositives).map { case (t, f) => t.toFloat / (t + f)}
    val recalls = numTruePositives.map(t => t.toFloat / numPositives)
    val fpr = numFalsePositives.map(f => f.toFloat / numNegatives)
    val rocCurve = Seq((0.0f, 0.0f)) ++ fpr.zip(recalls) ++ Seq((1.0f, 1.0f))

    validateMetrics(metrics, thresholds, rocCurve, precisions, recalls)
  }


  test("binary evaluation metrics  where all examples have positive label") {
    val predictions = Array(0.5, 0.5).map(Array(_)).toNDArray
    val real = Array(1.0, 1.0).map(Array(_)).toNDArray
    val metrics = new BinaryClassificationMetrics(real, predictions)

    val thresholds = Seq(0.5f)
    val precisions = Seq(1.0f)
    val recalls = Seq(1.0f)
    val fpr = Seq(0.0f)
    val rocCurve = Seq((0.0f, 0.0f)) ++ fpr.zip(recalls) ++ Seq((1.0f, 1.0f))

    validateMetrics(metrics, thresholds, rocCurve, precisions, recalls)
  }

  test("binary evaluation metrics  where all examples have negative label") {
    val predictions = Array(0.5, 0.5).map(Array(_)).toNDArray
    val real = Array(0.0, 0.0).map(Array(_)).toNDArray
    val metrics = new BinaryClassificationMetrics(real, predictions)

    val thresholds = Seq(0.5f)
    val precisions = Seq(0.0f)
    val recalls = Seq(0.0f)
    val fpr = Seq(1.0f)
    val rocCurve = Seq((0.0f, 0.0f)) ++ fpr.zip(recalls) ++ Seq((1.0f, 1.0f))

    validateMetrics(metrics, thresholds, rocCurve, precisions, recalls)
  }

  test("aucRoc random") {

    val real = Array(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0).map(Array(_)).toNDArray

    val predictionsRandom = Array( 0.5, 0.5 , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5).map(Array(_)).toNDArray
    val metricsRandom = new BinaryClassificationMetrics(real, predictionsRandom)
    assert(metricsRandom.aucRoc === 0.5f +- 0.05f)

    val predictionsNegative = Array(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0).map(Array(_)).toNDArray
    val metricsNegative = new BinaryClassificationMetrics(real, predictionsNegative)
    assert(metricsNegative.aucRoc === 0.0f +- 0.05f)

    val predictionsTrue = Array(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0).map(Array(_)).toNDArray
    val metricsTrue = new BinaryClassificationMetrics(real, predictionsTrue)
    assert(metricsTrue.aucRoc === 1.0f +- 0.05f)

  }

}
