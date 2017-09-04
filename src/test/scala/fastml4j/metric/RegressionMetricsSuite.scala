package fastml4j.metric

import java.util.Random

import fastml4j.regression.LinearRegression
import org.nd4s.Implicits._
import org.scalatest.Matchers._
import org.scalatest._

import math.random
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.DataGenerators._

class RegressionMetricsSuite extends FunSuite {


  test("perfect test") {
    val labels = (0 to 100).map( x => Array(x.toFloat)).toArray
    val output = labels.map(_.map(_ + math.random / 100))
    val evaluator = new RegressionMetrics(labels.toNDArray, output.toNDArray)

    assert(evaluator.rootMeanSquaredError === 0.0f +- 0.1f)
    assert(evaluator.meanSquaredError === 0.0f +- 0.1f)
    assert(evaluator.meanAbsoluteError === 0.5f +- 0.1f)
  }


  test("imperfect test") {
    val labels = (0 to 1000).map( x => Array(x)).toArray
    val output = labels.reverse.map(_.map( _  + math.random ))
    val evaluator = new RegressionMetrics(labels.toNDArray, output.toNDArray)

    assert(evaluator.rootMeanSquaredError === 500.0f +- 100.0f)
    assert(evaluator.meanSquaredError === 400000.0f +- 100000.0f )
    assert(evaluator.meanAbsoluteError === 500000.0f +- 10000.0f)

  }


}
