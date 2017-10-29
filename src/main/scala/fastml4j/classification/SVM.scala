package fastml4j.classification


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.optimizer._
import fastml4j.loss._
import org.nd4j.linalg.dataset.DataSet
import fastml4j.util.Implicits._
import fastml4j.util.Intercept

/**
  * Created by rzykov on 23/06/17.
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

    val optimizer: Optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
    //  case "GradientDescentDecreasingLearningRate" => new GradientDescentDecreasingLearningRate(maxIterations, alpha, eps)
      case "PegasosSGD" => new PegasosSGD(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
        new HingeLossL2(lambdaL2, calcIntercept),
        initWeights.getOrElse(Nd4j.zeros(dataSet.numInputs)),
        dataSet)

    intercept = extractIntercept(weightsOut)
    weights = extractWeights(weightsOut)
    losses = lossesOut
  }

  def predictClass(inputVector: INDArray): Float = {
    val sign = math.signum((inputVector dot weights.T).sumFloat)
    if( sign != 0 ) sign.toFloat else 1.0f
  }

  def predict(inputVector:  INDArray): Float = {
    (inputVector dot weights + intercept ).sumFloat
  }

}
