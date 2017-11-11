package com.github.rzykov.fastml4j.util

import org.scalatest._
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.scalatest.Matchers._
import org.nd4j.linalg.factory.Nd4j
import com.github.rzykov.fastml4j.util.DataGenerators._

/**
  * Created by rzykov on 02/07/17.
  */
class ImplicitSuite extends FunSuite with BeforeAndAfter {

  test("Intercept") {
    val features = Array(Array(1.0,2.0, 3.0), Array(4.0, 5.0, 6.0)).toNDArray
    val labels = Array(Array(10.0), Array(20.0)).toNDArray
    val dataSet = new DataSet(features, labels)
    val testDataSet = dataSet.addIntercept
    val dstFeatures = Array(Array(1.0,2.0, 3.0, 1.0), Array(4.0, 5.0, 6.0, 1.0)).toNDArray
    assert(testDataSet.getFeatures == dstFeatures)
  }

}
