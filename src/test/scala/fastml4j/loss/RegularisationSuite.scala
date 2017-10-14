package fastml4j.loss
import org.scalatest.Matchers._
import org.scalatest._

import math.random
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import fastml4j.util.Implicits._


class RegularisationSuite  extends FunSuite with BeforeAndAfter {

  test("L2") {
    val l2WithIntercept = L2(2, true)

    val l2WithoutIntercept = L2(2, false)

    val weights = Array(1.0, 2.0, 3.0).toNDArray

    assert(l2WithIntercept.lossRegularisation(weights) == 5.0f)
    assert(l2WithoutIntercept.lossRegularisation(weights) == 14.0f)

    assert(l2WithIntercept.gradientRegularisation(weights).sumFloat == 6.0f)
    assert(l2WithoutIntercept.gradientRegularisation(weights).sumFloat == 12.0f)

  }
}