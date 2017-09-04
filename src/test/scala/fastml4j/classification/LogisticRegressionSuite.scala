package fastml4j.classification

/**
  * Created by rzykov on 25/06/17.
  */

import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.rng.distribution.impl.BinomialDistribution
import fastml4j.util.DataGenerators._

import math.{random, exp}
import util.Random

/**
  * Created by rzykov on 01/07/17.
  */
class LogisticRegressionSuite extends FunSuite with BeforeAndAfter  {



  test("simple synthetic test") {
    val coef = 3.0f
    val intercept = 1.0f
    val samples = 100

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)
    val dataSet = new DataSet(points.toNDArray, labels.toNDArray)
    dataSet.validate()
    val lr = new LogisticRegression(lambdaL2 = 0.0f, maxIterations = 10000  ,alpha = 0.1f, eps = 1e-10f)
    lr.fit(dataSet)
    assert(lr.weights.getFloat(0,0) === coef +- 1)
    assert(lr.weights.getFloat(0,1) === intercept +- 1)

  }


}
