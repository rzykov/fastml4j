package mlscala.optimize

/**
  * Created by rzykov on 25/06/17.
  */
import mlscala.losses.OLSLoss
import mlscala.optimizer.GradientDescent
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


class OptimizeSuite extends FunSuite with BeforeAndAfter {
  test("Gradient descent") {
    val coef1 = 30.0
    val coef2 = 20.0

    val data = (1 to 8).map(Array(_,1.0)).toArray
    val out = data.map{ case Array(a, b) => coef1 * a + coef2 * b }

    val ols = new OLSLoss(0)
    val optimizer = new GradientDescent(maxIterations = 500, alpha = 0.05, eps = 1e-3)
    val (weights, losses) = optimizer.optimize(ols, Array(math.random,math.random).toNDArray, data.toNDArray, out.toNDArray )
    assert(weights.getDouble(0,0) === coef1 +- 1)
    assert(weights.getDouble(0,1) === coef2 +- 1)
    assert(losses.sliding(2).filter{ case Seq(left, right) => left < right}.size === 0 +- 5) // losses are descending

  }


}
