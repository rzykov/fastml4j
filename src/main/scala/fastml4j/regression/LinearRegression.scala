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
  * Created by rzykov on 02/07/17.
  */
class LinearRegression(
  val lambdaL2: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Float = 1e-6f,
  val standardize: Boolean = true,
  val intercept: Boolean = true) extends RegressionModel with Intercept {

  private class OLSLossL2 (override val lambdaL2: Float, override val intercept: Boolean)
    extends OLSLoss with L2 with Intercept

  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {

    dataSet.validate()

    val dataSetIntercept: DataSet = dataSetWithIntercept(dataSet)

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
      new OLSLossL2(lambdaL2, intercept),
      initWeights.getOrElse(Nd4j.zeros(dataSetIntercept.numInputs)),
      dataSetIntercept)

    interceptValue = extractIntercept(weightsOut)
    weights = extractWeights(weightsOut)
    losses = lossesOut
  }

  def predict(inputVector: INDArray): Float = {
    (inputVector dot weights).sumFloat
  }

  def transform(dataSet: DataSet): INDArray = (weights dot dataSet.getFeatures.T)

}
