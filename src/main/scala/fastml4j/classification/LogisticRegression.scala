package fastml4j.classification

/**
  * Created by rzykov on 13/07/17.
  */

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.optimizer._
import fastml4j.losses._
import fastml4j.util.Implicits._
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.ops.transforms.Transforms


/**
  * Created by rzykov on 13/07/17.
  */
class LogisticRegression(val lambdaL2: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Float = 1e-6f) extends ClassificationModel {

  var weights: INDArray = Nd4j.zeros(1)
  var losses: Seq[Float] = Seq[Float]()


  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {
    dataSet.validate()

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
      new LogisticLoss(lambdaL2),
      initWeights = initWeights.getOrElse(Nd4j.zeros(dataSet.numInputs)),
      dataset = dataSet)

    weights = weightsOut
    losses = lossesOut
  }

  def predictClass(inputVector: INDArray): Float = {
    math.round(predict(inputVector))
  }

  def predict(inputVector:  INDArray): Float = {
    Transforms.sigmoid(inputVector dot weights).sumT
  }

}
