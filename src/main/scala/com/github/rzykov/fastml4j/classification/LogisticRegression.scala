package com.github.rzykov.fastml4j.classification

import com.github.rzykov.fastml4j.loss.{HingeLoss, L2}
import com.github.rzykov.fastml4j.optimizer.GradientDescent
import com.github.rzykov.fastml4j.util.Intercept
import com.github.rzykov.fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import com.github.rzykov.fastml4j.optimizer._
import com.github.rzykov.fastml4j.loss._
import com.github.rzykov.fastml4j.util.Intercept
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * Logistic regression. Creates a class of the logistic regression model
  *
  * @param lambdaL2  - regularisation parameter for L2
  * @param alpha  - step parameter for optimizer
  * @param maxIterations - max iterations for optimizer
  * @param stohasticBatchSize - batch size, valid only for stohastic gradient descent
  * @param optimizerType - which optimizer to use
  * @param eps - minimum change for loss function, used by optimizer
  * @param calcIntercept - include fitting of the intercept
  */

class LogisticRegression
  (val lambdaL2: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Float = 1e-6f,
  val calcIntercept: Boolean = true) extends ClassificationModel with Intercept {

  private class LogisticLossL2(override val lambdaL2: Float, override val calcIntercept: Boolean)
    extends HingeLoss with L2 with Intercept

  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {
    dataSet.validate()

    val dataSetIntercept: DataSet = dataSetWithIntercept(dataSet)

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) =
      optimizer.optimize(
        new LogisticLossL2(lambdaL2, calcIntercept),
        initWeights = initWeights.getOrElse(Nd4j.zeros(dataSetIntercept.numInputs)),
        dataset = dataSetIntercept)

    intercept = extractIntercept(weightsOut)
    weights = extractWeights(weightsOut)
    losses = lossesOut
  }

  def predictClass(inputVector: INDArray): Float = {
    math.round(predict(inputVector))
  }

  def predict(inputVector:  INDArray): Float = {
    Transforms.sigmoid(inputVector dot weights + intercept).sumFloat
  }

}
