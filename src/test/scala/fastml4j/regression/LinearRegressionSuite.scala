package fastml4j.regression

import org.scalatest._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.Matchers._
import org.nd4j.linalg.factory.Nd4j

/**
  * Created by rzykov on 02/07/17.
  */
class LinearRegressionSuite extends FunSuite with BeforeAndAfter {

  test("Simple test") {
    val coef1 = 1.0
    val coef2 = 0.5

    val x = (0 to   500).map { x => Array(x.toDouble,1.0) }.toArray
    val y = x.map {case Array(x1, x2) => x1 * coef1 + x2 * coef2 + math.random/10 }

    val lr = new LinearRegression(lambdaL2 = 0.0, alpha = 0.000001, eps = 1e-4, maxIterations = 2000)
    lr.fit(x.toNDArray, y.toNDArray, Option(Nd4j.zeros(2)) )

    val weights = lr.weights
    assert(weights.getDouble(0,0) === coef1 +- 1)
    assert(weights.getDouble(0,1) === coef2 +- 1)
  }

}
