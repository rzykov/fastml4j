package fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */
import org.nd4s.Implicits._
import org.scalatest.Matchers._
import org.scalatest._

import math.random
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import fastml4j.util.Implicits._


class OLSSuite extends FunSuite with BeforeAndAfter {

  test("OLSLoss: Loss") {
    val trainData = Array(Array(-1,2,3),Array(-4,5,6))
    val labels = Array(Array(-1), Array(1))
    val weights = Array(0.2, 0.7, 0.9)

    val loss = new OLSLoss(NoRegularisation)
    assert( loss.loss(weights.toNDArray, new DataSet(trainData.toNDArray, labels.toNDArray)) === 18.6f +- 0.01f)
    assert( loss.loss(weights.toNDArray, new DataSet((trainData ++ trainData).toNDArray, (labels ++ labels).toNDArray)) === 18.6f +- 0.01f)

    val loss2 = new OLSLoss(L2(1))
    assert( loss2.loss(weights.toNDArray, new DataSet(trainData.toNDArray, labels.toNDArray)) === 19.27f +- 0.01f)}

  test("OLSLoss: gradient") {

    val trainData2 = Array(Array(-1,2,3),Array(-4,5,6)).toNDArray
    val labels2 = Array(Array(-1), Array(1)).toNDArray
    val weights2 = Array(0.2, 0.7, 0.9).toNDArray

    val loss2 = new OLSLoss(NoRegularisation)
    assert(loss2.numericGradient(weights2, new DataSet(trainData2, labels2), 1E-3f).sumT ===
      loss2.gradient(weights2, new DataSet(trainData2, labels2)).sumT +- 0.3f)

    val trainData = Array[Array[Double]](Array(-1.0,2.0,3.0),Array(-4.0,5.0,6.0), Array(-2.0,5.0,4.0)).toNDArray
    val labels = Array(Array(-1.0), Array(1.0), Array(1.0)).toNDArray
    val weights = Array[Double](0.2, 0.7, 0.9).toNDArray

    val loss = new OLSLoss(NoRegularisation)
    assert(loss.numericGradient(weights, new DataSet(trainData, labels), 1E-3f).sumT[Float] ===
      loss.gradient(weights, new DataSet(trainData, labels)).sumT[Float] +- 0.3f)
  }

  test("OLSLoss: gradient random") {

    val coef1 = 10
    val coef2 = 3
    val intercept = 4
    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- (1 to samples)
            a = random * 10 - 5
            b = random * 100 - 50} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => Array(a * coef1 + b * coef2 + c * intercept + random) }

    val weights: Seq[Array[Double]] =  (1 to 100).map{ _ => Array(random *10 - 5, random * 2, random - 0.5 ) }

    val loss2 = new OLSLoss(NoRegularisation)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toArray.toNDArray)).sumT,
      loss2.numericGradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toArray.toNDArray)).sumT)}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05f)
  }

}
