package fastml4j.classification

/**
  * Created by rzykov on 25/06/17.
  */

import fastml4j.loss.HingeLoss
import org.scalatest._
import org.scalatest.Matchers._
import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.DataGenerators.generateLogisticInput

import math.random
import scala.util.Random


/**
  * Created by rzykov on 01/07/17.
  */
class SVMSuite extends FunSuite with BeforeAndAfter  {

  test("simple synthetic test") {
    val coef = 3.0f
    val intercept = 1.0f
    val samples = 10000

    val (points, labels) = generateLogisticInput(intercept, coef, samples, 100)

    val lr = new SVM(regularisationFactor = 0.0f, maxIterations = 10000  ,alpha = 0.001f, eps = 1e-4f)
    lr.fit(new DataSet(points.toNDArray, labels.toNDArray))
    assert(lr.weights.get(0,0) === coef +- 1)
    assert(lr.interceptValue === intercept +- 1)
  }


}
