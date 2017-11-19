package com.github.rzykov.fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */

import org.scalatest._
import org.scalatest.Matchers._
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.math.random


class LogisticLossSuite extends FunSuite with BeforeAndAfter {

  test("LogisticLoss: one sample test") {
    val trainData = Array(Array(-1,2,1)).toNDArray
    val labels = Array(Array(1.0)).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new LogisticLoss
    assert( loss.loss(weights, new DataSet(trainData, labels)) === 0.12f +- 0.01f)}

  test("LogisticLoss: gradient checking by random") {

    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- 1 to samples
                                              a = random * 1
                                              b = random * 1 - 0.5} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => if (a > b )  Array(1.0) else Array(0.0) }.toArray

    val weights: Seq[Array[Double]] =  (1 to 10).map{ _ => Array(random *2 - 2, random , random - 0.5 ) }

    val loss2 = new LogisticLoss

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toNDArray)).sumFloat,
      loss2.numericGradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toNDArray)).sumFloat)}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.1f)
  }

}
