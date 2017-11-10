package fastml4j.regression

import fastml4j.loss.{L2, OLSLoss}
import fastml4j.optimizer.GradientDescent
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import fastml4j.util.Implicits._
import fastml4j.util.Intercept


/**
  * Linear regression model based on OLS (ordinary least squares)
  *
  * @param lambdaL2  - regularisation parameter for L2
  * @param alpha  - step parameter for optimizer
  * @param maxIterations - max iterations for optimizer
  * @param stohasticBatchSize - batch size, valid only for stohastic gradient descent
  * @param optimizerType - which optimizer to use
  * @param eps - minimum change for loss function, used by optimizer
  * @param calcIntercept - include fitting of the intercept
  */

class LinearRegression(
  val lambdaL2: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Float = 1e-6f,
  val calcIntercept: Boolean = true) extends RegressionModel with Intercept {

  private class OLSLossL2 (override val lambdaL2: Float, override val calcIntercept: Boolean)
    extends OLSLoss with L2 with Intercept

  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {

    dataSet.validate()

    val dataSetIntercept: DataSet = dataSetWithIntercept(dataSet)

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
      new OLSLossL2(lambdaL2, calcIntercept),
      initWeights.getOrElse(Nd4j.zeros(dataSetIntercept.numInputs)),
      dataSetIntercept)

    intercept = extractIntercept(weightsOut)
    weights = extractWeights(weightsOut)
    losses = lossesOut
  }

  def predict(inputVector: INDArray): Float = {
    (inputVector dot weights).sumFloat
  }

  def transform(dataSet: DataSet): INDArray = (weights dot dataSet.getFeatures.T)

}
