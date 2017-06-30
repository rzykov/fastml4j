package mlscala.loss

/**
  * Created by rzykov on 25/06/17.
  */
import mlscala.losses.{HingeLoss, OLSLoss}
import org.nd4s.Implicits._
import org.scalatest.Matchers._
import org.scalatest._


class OLSSuite extends FunSuite with BeforeAndAfter {

  test("OLSLoss: Loss") {
    val trainData = Array(Array(-1,2,3),Array(-4,5,6)).toNDArray
    val labels = Array(-1, 1).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new OLSLoss(0)
    assert( loss.loss(weights, trainData, labels) === 18.6 +- 0.01)

    val loss2 = new OLSLoss(1)
    assert( loss2.loss(weights, trainData, labels) === 19.27 +- 0.01)}

  test("OLSLoss: gradient") {

    val trainData2 = Array(Array(-1,2,3),Array(-4,5,6)).toNDArray
    val labels2 = Array(-1, 1).toNDArray
    val weights2 = Array(0.2, 0.7, 0.9).toNDArray

    val loss2 = new OLSLoss(0)
    assert(loss2.numericGradient(weights2, trainData2, labels2, 1E-3).sumT[Double] ===
      loss2.gradient(weights2, trainData2, labels2).sumT[Double] +- 0.3)

    val trainData = Array[Array[Double]](Array(-1.0,2.0,3.0),Array(-4.0,5.0,6.0), Array(-2.0,5.0,4.0)).toNDArray
    val labels = Array[Double](-1.0, 1.0, 1.0).toNDArray
    val weights = Array[Double](0.2, 0.7, 0.9).toNDArray

    val loss = new OLSLoss(0)
    assert(loss.numericGradient(weights, trainData, labels, 1E-3).sumT[Double] ===
      loss.gradient(weights, trainData, labels).sumT[Double] +- 0.3)

  }

}
