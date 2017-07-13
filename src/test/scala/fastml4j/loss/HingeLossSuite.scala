package fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */
import fastml4j.losses.HingeLoss
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.math.random


class HingeLossSuite extends FunSuite with BeforeAndAfter {

  test("HingeLoss: one sample test") {
    val trainData = Array(Array(-1,2,3)).toNDArray
    val labels = Array(-1).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new HingeLoss(0)
    assert( loss.loss(weights, trainData, labels) === 4.9 +- 0.01)}


  test("HingeLoss: Loss") {
    val trainData = Array(Array(-1,2,3),Array(-4,5,6)).toNDArray
    val labels = Array(-1, 1).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new HingeLoss(0)
    assert( loss.loss(weights, trainData, labels) === 2.45 +- 0.01)

    val loss2 = new HingeLoss(1)
    assert( loss2.loss(weights, trainData, labels) === 3.11 +- 0.01)}

  test("HingeLoss: gradient random") {

    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- (1 to samples)
                                              a = random * 10 - 5
                                              b = random * 100 - 50} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => if (a > b )  1.0 else -1.0 }

    val weights: Seq[Array[Double]] =  (1 to 100).map{ _ => Array(random *10 - 5, random * 2, random - 0.5 ) }

    val loss2 = new HingeLoss(0)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double],
      loss2.numericGradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double])}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05)
  }





}
