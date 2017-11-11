package com.github.rzykov.fastml4j.loss

import com.github.rzykov.fastml4j.loss.{L2, Loss}
import org.scalatest.Matchers._
import org.scalatest._

import math.random
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import com.github.rzykov.fastml4j.util.Implicits._
import com.github.rzykov.fastml4j.util.Intercept


class RegularisationSuite  extends FunSuite with BeforeAndAfter {


  class L2test extends Loss  {
    override def loss(weights: INDArray, dataSet: DataSet): Float = {
      0
    }
    override def gradient(weights: INDArray, dataSet: DataSet): INDArray =
      weights
  }

  private class L2testLoss(override val lambdaL2: Float, override val calcIntercept: Boolean)
    extends L2test with L2 with Intercept


  test("L2") {
    val l2WithIntercept = new L2testLoss(2, true)

    val l2WithoutIntercept = new L2testLoss(2, false)

    val weights = Array(1.0, 2.0, 3.0).toNDArray

    val trainData = Array(Array(-1,2,1)).toNDArray
    val labels = Array(Array(1.0)).toNDArray

    val dataSet = new DataSet(trainData, labels)

    assert(l2WithIntercept.loss(weights, dataSet) == 5.0f)
    assert(l2WithoutIntercept.loss(weights, dataSet) == 14.0f)

    assert(l2WithIntercept.gradient(weights, dataSet).toArray.deep == Array(Array(3.0f, 6.0f, 3.0f)).deep)
    assert(l2WithoutIntercept.gradient(weights, dataSet).toArray.deep == Array(Array(3.0f, 6.0f, 9.0f)).deep)
  }
}