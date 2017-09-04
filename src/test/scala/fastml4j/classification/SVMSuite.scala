package fastml4j.classification

/**
  * Created by rzykov on 25/06/17.
  */

import fastml4j.losses.HingeLoss
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import math.random
import scala.util.Random


/**
  * Created by rzykov on 01/07/17.
  */
class SVMSuite extends FunSuite with BeforeAndAfter  {

  //taken from Spark test
  def generateLogisticInput(
    offset: Float,
    scale: Float,
    nPoints: Int,
    seed: Int): (Array[Array[Float]], Array[Array[Float]]) = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Float](nPoints)(rnd.nextGaussian().toFloat)

    val y = (0 until nPoints).map { i =>
      val p = 1.0f / (1.0f + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextFloat() < p) Array(1.0f) else Array(-1.0f)}
      .toArray

    val features = x1.map(Array(_, 1.0f))
    (features, y)
  }

  test("simple synthetic test") {
    val coef = 3.0f
    val intercept = 1.0f
    val samples = 10000

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)

    val lr = new SVM(lambdaL2 = 0.0f, maxIterations = 10000  ,alpha = 0.001f, eps = 1e-4f)
    lr.fit(new DataSet(points.toNDArray, labels.toNDArray))
    assert(lr.weights.getFloat(0,0) === coef +- 1)
    assert(lr.weights.getFloat(0,1) === intercept +- 1)
  }


}
