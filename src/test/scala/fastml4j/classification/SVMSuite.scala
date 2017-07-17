package fastml4j.classification

/**
  * Created by rzykov on 25/06/17.
  */

import fastml4j.losses.HingeLoss
import org.scalatest._
import org.scalatest.Matchers._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import math.random
import scala.util.Random


/**
  * Created by rzykov on 01/07/17.
  */
class SVMSuite extends FunSuite with BeforeAndAfter  {

  //taken from Spark test
  def generateLogisticInput(
    offset: Double,
    scale: Double,
    nPoints: Int,
    seed: Int): (Array[Array[Double]], Array[Double]) = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextDouble() < p) 1.0 else -1.0}
      .toArray

    val features = x1.map(Array(_, 1.0))
    (features, y)
  }

  test("Hinge Loss: test"){
    //https://stats.stackexchange.com/questions/4608/gradient-of-hinge-loss
    val trainData:Array[Array[Double]] = Array(Array(0,1),Array(1,2),Array(3,0),Array(4,1),Array(1,1))
    val labels = Array(1,1,-1,-1,-1).toNDArray
    val weights = Array(-0.326, 0.226).toNDArray
    val loss = new HingeLoss(0)

    val lr = new SVM(lambdaL2 = 0.0, maxIterations = 10000  ,alpha = 0.1, eps = 1e-3)
    lr.fit(trainData.toNDArray, labels)
    println( lr.weights)
    println( lr.losses.size)
    println( lr.losses.take(10))
    println( lr.losses.takeRight(10))

  }

  test("simple synthetic test") {
    val coef = 2.0
    val intercept = 1
    val samples = 100

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)


    val lr = new SVM(lambdaL2 = 0.0, maxIterations = 10000  ,alpha = 0.001, eps = 1e-3)
    lr.fit(points.toNDArray, labels.toNDArray)
    println( lr.weights)
    println( lr.losses.size)
    println( lr.losses.take(10))
    println( lr.losses.takeRight(10))
  }


}
