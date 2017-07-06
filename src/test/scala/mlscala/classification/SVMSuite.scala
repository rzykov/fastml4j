package mlscala.classification

/**
  * Created by rzykov on 25/06/17.
  */

import mlscala.losses.HingeLoss
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import math.random


/**
  * Created by rzykov on 01/07/17.
  */
class SVMSuite extends FunSuite with BeforeAndAfter  {

  test("2 lines") {

    val pointsLowerLine: Array[Array[Double]] = (0 to 100).map{ x =>  Array(5 * x + 3 + random / 20, random) }.toArray
    val pointsUpperLine: Array[Array[Double]] = (0 to 100).map{ x =>  Array(5 * x + 1 + random / 20, random) }.toArray
    val points = pointsLowerLine ++ pointsUpperLine

    val labels: Array[Double] = pointsLowerLine.map( x => -1.0) ++ pointsUpperLine.map( x => 1.0)

    val svm = new SVM(lambdaL2 = 0.0, maxIterations = 1000  ,alpha = 0.00005, eps = 1e-8)
    svm.fit(points.toNDArray, labels.toNDArray)
    println( svm.weights)
    println( svm.losses.size)
    println( svm.losses.take(10))
  }

  test("HingeLoss: gradient random") {

    val coef1 = 10
    val coef2 = 3
    val intercept = 4
    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- (1 to 100)
                                              a = random * 10 - 5
                                              b = random * 100 - 50} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => a * coef1 + b * coef2 + c * intercept + random }

    val weights: Seq[Array[Double]] =  (1 to 100).map{ _ => Array(random *10 - 5, random * 2, random - 0.5 ) }

    val loss2 = new HingeLoss(0)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double],
      loss2.numericGradient(w.toNDArray, trainData.toArray.toNDArray, labels.toNDArray).sumT[Double])}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05)
  }


}
