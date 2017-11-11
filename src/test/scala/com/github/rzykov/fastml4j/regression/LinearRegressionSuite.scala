package com.github.rzykov.fastml4j.regression

import com.github.rzykov.fastml4j.regression.LinearRegression
import org.scalatest._
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.scalatest.Matchers._
import org.nd4j.linalg.factory.Nd4j
import com.github.rzykov.fastml4j.util.DataGenerators._

/**
  * Created by rzykov on 02/07/17.
  */
class LinearRegressionSuite extends FunSuite with BeforeAndAfter {

  test("with data generators: 1 coef without intercept") {
    val coef1 = 5f
    val dataset = generateLinearInput(0.0f, Array(coef1), Array(0.9f), Array(0.7f), 100, 42, 0.1f)
    val lr = new LinearRegression(lambdaL2 = 0.0f, alpha = 0.1f, eps = 1e-4f, maxIterations = 2000, calcIntercept=false)
    lr.fit(dataset)
    val weights = lr.weights

    assert(weights.getFloat(0,0) === coef1 +- 0.2f)
  }

  test("with data generators: 2 coef without intercept") {
    val coef1 = 5f
    val coef2 = 7f

    val dataset = generateLinearInput(0.0f, Array(coef1, coef2), Array(0.9f, 1.5f), Array(0.7f, 0.5f), 100, 42, 0.1f)
    val lr = new LinearRegression(lambdaL2 = 0.0f, alpha = 0.1f, eps = 1e-4f, maxIterations = 2000, calcIntercept=false)
    lr.fit(dataset)
    val weights = lr.weights
    assert(weights.getFloat(0,0) === coef1 +- 0.2f)
    assert(weights.getFloat(0,1) === coef2 +- 0.2f)
  }

  test("with data generators: 2 coef with intercept") {
    val coef1 = 5f
    val coef2 = 7f
    val intercept = 1f

    val dataset = generateLinearInput(intercept, Array(coef1, coef2), Array(0.9f, 1.5f), Array(0.7f, 0.5f), 1000, 42, 0.1f)
    val lr = new LinearRegression(lambdaL2 = 0.0f, alpha = 0.1f, eps = 1e-4f, maxIterations = 2000, calcIntercept=true)
    lr.fit(dataset)
    val weights = lr.weights
    assert(weights.getFloat(0,0) === coef1 +- 0.2f)
    assert(weights.getFloat(0,1) === coef2 +- 0.2f)
    assert(lr.intercept === intercept +- 0.3f)
  }

}
