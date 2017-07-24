package fastml4j.loss

/**
  * Created by rzykov on 25/06/17.
  */
import fastml4j.losses.HingeLoss
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
    val labels = Array(Array(-1.0)).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new HingeLoss(0)
    assert(loss.loss(weights, new DataSet(trainData, labels)) === 4.9 +- 0.01)
    assert(loss.gradient(weights, new DataSet(trainData, labels)) == Array(Array(-1.0,2.0,3.0)).toNDArray)
  }

  test("Hinge Loss: test"){
    //https://stats.stackexchange.com/questions/4608/gradient-of-hinge-loss
    val trainData:Array[Array[Double]] = Array(Array(0,1),Array(1,2),Array(3,0),Array(4,1),Array(1,1))
    val labels = Array(1,1,-1,-1,-1).map(Array(_)).toNDArray
    val weights = Array(-0.326, 0.226).toNDArray
    val loss = new HingeLoss(0)
    println(loss.loss(weights, new DataSet(trainData.toNDArray, labels)))
    println(loss.gradient(weights, new DataSet(trainData.toNDArray, labels)))

  }

 test("HingeLoss: Loss") {
    val trainData = Array(Array(-1,2,3),Array(-4,5,6)).toNDArray
    val labels = Array(Array(-1), Array(1)).toNDArray
    val weights = Array(0.2, 0.7, 0.9).toNDArray

    val loss = new HingeLoss(0)
    assert(loss.loss(weights, new DataSet(trainData, labels)) === 2.45 +- 0.01)

    val loss2 = new HingeLoss(1)
    assert(loss2.loss(weights, new DataSet(trainData, labels)) === 3.11 +- 0.01)}

  test("HingeLoss: gradient random") {
    val samples = 1000
    val trainData: Seq[Array[Double]] = for { i <- (1 to samples)
                                              a = random * 10 - 5
                                              b = random * 100 - 50} yield Array(a ,b, 1.0)

    val labels = trainData.map{ case Array(a, b, c) => if (a > b )  Array(1.0) else Array(-1.0) }
    val weights: Seq[Array[Double]] =  (1 to 100).map{ _ => Array(random *10 - 5, random * 2, random - 0.5 ) }
    val loss2 = new HingeLoss(0)

    val gradients = weights.map{ w =>   (loss2.gradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toArray.toNDArray)).sumT[Double],
      loss2.numericGradient(w.toNDArray, new DataSet(trainData.toArray.toNDArray, labels.toArray.toNDArray)).sumT[Double])}
      .map{ case(grad, nGrad ) => (grad - nGrad)/grad  }

    assert( (gradients.sum / gradients.size) < 0.05)
  }

}
