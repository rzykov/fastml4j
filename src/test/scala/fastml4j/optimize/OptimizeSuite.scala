package fastml4j.optimize

/**
  * Created by rzykov on 25/06/17.
  */
import fastml4j.losses.{LogisticLoss, OLSLoss}
import fastml4j.optimizer.{GradientDescent, PegasosSGD}
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.util.Random


class OptimizeSuite extends FunSuite with BeforeAndAfter {

  test("Gradient descent") {
    val coef1 = 2.0
    val coef2 = 1.0

    val data = (1 to 1000).map(Array(_,1.0)).toArray
    val out = data.map{ case Array(a, b) => coef1 * a + coef2 * b + math.random / 10 }

    val ols = new OLSLoss(0)
    val optimizer = new GradientDescent(maxIterations = 1000, stepSize = 0.000005, eps = 1e-4)
    val (weights, losses) = optimizer.optimize(ols, Array(0.0,0.0).toNDArray, data.toNDArray, out.toNDArray )
    println(losses.size)

  //  assert(losses.sliding(2).filter{ case Seq(left, right) => left < right}.size === 0 +- 5) // losses are descending
    assert(weights.getDouble(0,0) === coef1 +- 1)
    assert(weights.getDouble(0,1) === coef2 +- 1)
  }

  //taken from Spark test
  def generateLogisticInput(
    offset: Double,
    scale: Double,
    nPoints: Int,
    seed: Int): (Array[Array[Double]], Array[Double]) = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextDouble() < p) 1.0 else 0.0}
      .toArray

    val features = x1.map(Array(_, 1.0))
    (features, y)
  }

  test("Pegasos SGD") {
    val coef = 3.0
    val intercept = 1.0
    val samples = 1000

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)

    val ols = new LogisticLoss(0)
    val optimizer = new PegasosSGD(maxIterations = 1000, lambda = 0.01, eps = 1e-4)
    val (weights, losses) = optimizer.optimize(ols, Array(0.0,0.0).toNDArray, points.toNDArray, labels.toNDArray )
    println(losses.size)

    //  assert(losses.sliding(2).filter{ case Seq(left, right) => left < right}.size === 0 +- 5) // losses are descending
    assert(weights.getDouble(0,0) === coef +- 1)
    assert(weights.getDouble(0,1) === intercept +- 1)
  }


}
