package fastml4j.optimize

/**
  * Created by rzykov on 25/06/17.
  */
import fastml4j.loss.{LogisticLoss, OLSLoss, NoRegularisation}
import fastml4j.optimizer.{GradientDescent, PegasosSGD}
import fastml4j.util.Implicits._
import org.scalatest._
import org.scalatest.Matchers._

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j


import scala.util.Random


class OptimizeSuite extends FunSuite with BeforeAndAfter {

  test("Gradient descent") {
    val coef1 = 2.0f
    val coef2 = 1.0f

    val data = (1 to 1000).map(Array(_,1.0)).toArray
    val out = data.map{ case Array(a, b) => coef1 * a + coef2 * b + math.random / 10 }

    val ols = new OLSLoss(NoRegularisation)
    val optimizer = new GradientDescent(maxIterations = 1000, stepSize = 0.000005f, eps = 1e-4f)
    val (weights, losses) = optimizer.optimize(ols, Array(0.0,0.0).toNDArray, new DataSet(data.toNDArray, out.toNDArray))

    assert(weights.getFloat(0,0) === coef1 +- 1)
    assert(weights.getFloat(0,1) === coef2 +- 1)
  }

  //taken from Spark test
  def generateLogisticInput(
    offset: Float,
    scale: Float,
    nPoints: Int,
    seed: Int): (Array[Array[Float]], Array[Array[Float]]) = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Float](nPoints)(rnd.nextGaussian() toFloat)

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextFloat() < p) Array(1.0f) else Array(0.0f)}
      .toArray

    val features = x1.map(Array(_, 1.0f))
    (features, y)
  }

  test("Pegasos SGD") {
    val coef = 3.0f
    val intercept = 1.0f
    val samples = 1000

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)

    val ols = new LogisticLoss(NoRegularisation)
    val optimizer = new PegasosSGD(maxIterations = 1000, lambda = 0.01f, eps = 1e-4f)
    val (weights, losses) = optimizer.optimize(ols, Array(0.0,0.0).toNDArray, new DataSet(points.toNDArray, labels.toNDArray))

    assert(weights.getFloat(0,0) === coef +- 1)
    assert(weights.getFloat(0,1) === intercept +- 1)
  }
}
