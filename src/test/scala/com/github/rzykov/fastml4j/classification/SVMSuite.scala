package com.github.rzykov.fastml4j.classification

/**
  * Created by rzykov on 25/06/17.
  */

import com.github.rzykov.fastml4j.loss.HingeLoss
import org.scalatest._
import org.scalatest.Matchers._
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import com.github.rzykov.fastml4j.util.DataGenerators.generateLogisticInput

import math.random
import scala.util.Random


/**
  * Created by rzykov on 01/07/17.
  */
class SVMSuite extends FunSuite with BeforeAndAfter  {

  //TODO Fix tests after normalization
  /*test("simple synthetic test") {
    val coef = 5f
    val weights = Array(1f*coef, 1f)
    val intercept = 0.0f
    val samples = 300
    val (points, labels) = generateLogisticInput(intercept, weights, samples, 100)
    val dataSet = new DataSet(points.toNDArray, labels.toNDArray)
    dataSet.validate()
    val lr = new SVM(lambdaL2 = 0.0f, maxIterations = 1000  ,alpha = 0.4f, eps = 1e-2f, calcIntercept = false)
    lr.fit(dataSet)

    assert(lr.weights.get(0,0) / lr.weights.get(0,1)  === coef +- 2.0f)
    assert(lr.intercept === 0f)
  }*/


}
