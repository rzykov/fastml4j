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


import math.{random, exp}
import util.Random

/**
  * Created by rzykov on 01/07/17.
  */
class LogisticRegressionSuite extends FunSuite with BeforeAndAfter  {

  //taken from Spark test
  def generateLogisticInput(
    offset: Double,
    scale: Double,
    nPoints: Int,
    seed: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextDouble() < p) Array(1.0) else Array(0.0)}
    .toArray

    val features = x1.map(Array(_, 1.0))
    (features, y)
  }

  test("simple synthetic test") {
    val coef = 3.0
    val intercept = 1.0
    val samples = 100

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)

    val lr = new LogisticRegression(lambdaL2 = 0.0, maxIterations = 10000  ,alpha = 0.1, eps = 1e-10)
    lr.fit(new DataSet(points.toNDArray, labels.toNDArray))
    assert(lr.weights.getDouble(0,0) === coef +- 1)
    assert(lr.weights.getDouble(0,1) === intercept +- 1)

  }


}
