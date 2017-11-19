package com.github.rzykov.fastml4j.classification

import com.github.rzykov.fastml4j.loss.{HingeLoss, L2}
import com.github.rzykov.fastml4j.optimizer.{GradientDescent, Optimizer, PegasosSGD}
import com.github.rzykov.fastml4j.util.Intercept
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import com.github.rzykov.fastml4j.optimizer._
import com.github.rzykov.fastml4j.loss._
import org.nd4j.linalg.dataset.DataSet
import com.github.rzykov.fastml4j.util.Implicits._
import com.github.rzykov.fastml4j.util.Intercept
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * SVM linear based on hinge loss. Creates a class of the SVM model
  *
  * @param lambdaL2  - regularisation parameter for L2
  * @param alpha  - step parameter for optimizer
  * @param maxIterations - max iterations for optimizer
  * @param stohasticBatchSize - batch size, valid only for stohastic gradient descent
  * @param optimizerType - which optimizer to use
  * @param eps - minimum change for loss function, used by optimizer
  * @param calcIntercept - include fitting of the intercept
  */

class SVM(val lambdaL2: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "PegasosSGD",
  val eps: Float = 1e-6f,
  val calcIntercept: Boolean = true) extends ClassificationModel with Intercept {

  private class HingeLossL2(override val lambdaL2: Float, override val calcIntercept: Boolean)
    extends HingeLoss with L2 with Intercept

  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {
    dataSet.validate()

    val dataSetIntercept: DataSet = dataSetWithIntercept(dataSet)

    val optimizer: Optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case "PegasosSGD" => new PegasosSGD(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
        new HingeLossL2(lambdaL2, calcIntercept),
        initWeights.getOrElse(Nd4j.zeros(dataSetIntercept.numInputs)),
        dataSetIntercept)

    intercept = extractIntercept(weightsOut)
    weights = extractWeights(weightsOut)
    losses = lossesOut
  }

  def predictClass(dataSet:  DataSet): INDArray = {
    val out = Transforms.sign(predict(dataSet))
    BooleanIndexing.replaceWhere(out, 1.0f, Conditions.equals(0.0f))
    out
  }

  def predictClass(inputVector: INDArray): Float = {
    val sign = math.signum((inputVector dot weights.T).sumFloat)
    if( sign == 0 ) 1.0f else sign.toFloat
  } //TODO - tests!

  def predict(inputVector:  INDArray): Float =
    (inputVector dot weights.T + intercept ).sumFloat

  def predict(dataSet:  DataSet): INDArray =
    dataSet.getFeatures dot weights.T + intercept

}
