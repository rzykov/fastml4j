package fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */
import fastml4j.losses.{HingeLoss, OLSLoss}
import org.nd4s.Implicits._
import org.scalatest.Matchers._
import org.scalatest._
import math.random
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray

class OLSSuite extends FunSuite with BeforeAndAfter {

  test("OLSLoss: Loss") {
    val trainData = Array(Array(-1,2,3),Array(-4,5,6))
    val labels = Array(-1, 1)
    val weights = Array(0.2, 0.7, 0.9)

    val loss = new OLSLoss(0)
    assert( loss.loss(weights.toNDArray, trainData.toNDArray, labels.toNDArray) === 18.6 +- 0.01)
    assert( loss.loss(weights.toNDArray, (trainData ++ trainData).toNDArray, (labels ++ labels).toNDArray) === 18.6 +- 0.01)


    val loss2 = new OLSLoss(1)
    assert( loss2.loss(weights.toNDArray, trainData.toNDArray, labels.toNDArray) === 19.27 +- 0.01)}

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

  test("OLSLoss: gradient random") {

    val coef1 = 10
    val coef2 = 3
    val intercept = 4
    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- (1 to samples)
            a = random * 10 - 5
            b = random * 100 - 50} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => a * coef1 + b * coef2 + c * intercept + random }

    val weights: Seq[Array[Double]] =  (1 to 100).map{ _ => Array(random *10 - 5, random * 2, random - 0.5 ) }

    val loss2 = new OLSLoss(0)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double],
      loss2.numericGradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double])}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05)
  }

}
