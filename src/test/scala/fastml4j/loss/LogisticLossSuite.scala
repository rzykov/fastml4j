package fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */
import fastml4j.losses.LogisticLoss
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.math.random


class LogisticLossSuite extends FunSuite with BeforeAndAfter {

  test("LogisticLoss: one sample test") {
    val trainData = Array(Array(-1,2,1)).toNDArray
    val labels = Array(1.0).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new LogisticLoss(0)
    assert( loss.loss(weights, trainData, labels) === 0.12 +- 0.01)}

  test("LogisticLoss: gradient checking by random") {

    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- (1 to samples)
                                              a = random * 1
                                              b = random * 1 - 0.5} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => if (a > b )  1.0 else 0.0 }

    val weights: Seq[Array[Double]] =  (1 to 10).map{ _ => Array(random *2 - 2, random , random - 0.5 ) }

    val loss2 = new LogisticLoss(0)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double],
      loss2.numericGradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double])}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05)
  }

}
