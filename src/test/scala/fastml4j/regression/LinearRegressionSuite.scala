package fastml4j.regression

import org.scalatest._
import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.scalatest.Matchers._
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.DataGenerators._

/**
  * Created by rzykov on 02/07/17.
  */
class LinearRegressionSuite extends FunSuite with BeforeAndAfter {

  test("Simple test") {
    val coef1 = 1.0f
    val coef2 = 0.5f

    val x = (0 to   500).map { x => Array(x.toFloat,1.0) }.toArray
    val y = x.map {case Array(x1, x2) => Array(x1 * coef1 + x2 * coef2 + math.random/10) }
    val dataSet = new DataSet(x.toNDArray, y.toNDArray)

    val lr = new LinearRegression(regularisationFactor = 0.0f, alpha = 0.000001f, eps = 1e-4f, maxIterations = 2000)
    lr.fit(dataSet)

    val weights = lr.weights
    assert(weights.getFloat(0,0) === coef1 +- 1)
    assert(weights.getFloat(0,1) === coef2 +- 1)
  }
/*
  test("with data generators") {
    val coef1 = 4.7
    val coef2 = 7.2


    val dataset = generateLinearInput(1.0, Array(coef1, coef2), Array(0.9, -1.3), Array(0.7, 1.2), 100, 42, 0.1)

    val lr = new LinearRegression(lambdaL2 = 0.0, alpha = 0.000001, eps = 1e-4, maxIterations = 2000)
    lr.fit(dataset)
    val weights = lr.weights
    println(weights)
    assert(weights.getFloat(0,0) === coef1 +- 0.1)
    assert(weights.getFloat(0,1) === coef2 +- 0.1)


  }*/

}
