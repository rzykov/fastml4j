package fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */

import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j


import scala.math.random


class HingeLossSuite extends FunSuite with BeforeAndAfter {

  test("HingeLoss: one sample test") {
    val trainData = Array(Array(-1.0, 2.0, 3.0)).toNDArray
    val labels = Array(Array(0.0)).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new HingeLoss(NoRegularisation)
    assert(loss.loss(weights, new DataSet(trainData, labels)) === 4.9f +- 0.01f)
    assert(loss.gradient(weights, new DataSet(trainData, labels)) == Array(Array(-1.0,2.0,3.0)).toNDArray)
  }

 test("HingeLoss: Loss") {
    val trainData = Array(Array(-1,2,3),Array(-4,5,6)).toNDArray
    val labels = Array(Array(0), Array(1)).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new HingeLoss(NoRegularisation)
    assert(loss.loss(weights, new DataSet(trainData, labels)) === 2.45f +- 0.01f)

    val loss2 = new HingeLoss(L2(1))
    assert(loss2.loss(weights, new DataSet(trainData, labels)) === 3.11f +- 0.01f)}

  test("HingeLoss: gradient random") {
    val samples = 1000
    val trainData: Seq[Array[Float]] = for { i <- (1 to samples)
                                              a = random * 10 - 5
                                              b = random * 100 - 50} yield Array(a.toFloat ,b.toFloat, 1.0f)

    val labels = trainData.map{ case Array(a, b, c) => if (a > b )  Array(1.0f) else Array(0.0f) }
    val weights: Seq[Array[Float]] =  (1 to 100).map{ _ => Array(random *10 - 5 toFloat, random * 2 toFloat, random - 0.5 toFloat ) }
    val loss2 = new HingeLoss(NoRegularisation)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toArray.toNDArray)).sumT,
      loss2.numericGradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toArray.toNDArray)).sumT)}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05f)
  }

}
