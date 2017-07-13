package fastml4j.classification

/**
  * Created by rzykov on 13/07/17.
  */

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.optimizer._
import fastml4j.losses._
import org.nd4j.linalg.ops.transforms.Transforms
/**
  * Created by rzykov on 13/07/17.
  */
class LogisticRegression(val lambdaL2: Double,
  val alpha: Double = 0.01,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Double = 1e-6) extends ClassificationModel {

  var weights: INDArray = Nd4j.zeros(1)
  var losses: Seq[Double] = Seq[Double]()


  def fit(trainData: INDArray, labels: INDArray, initWeights: Option[INDArray] = None) = {

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
      new LogisticLoss(lambdaL2),
      initWeights = initWeights.getOrElse(Nd4j.zeros(trainData.columns)),
      trainData = trainData,
      labels = labels)

    weights = weightsOut
    losses = lossesOut
  }

  def predictClass(inputVector: INDArray): Double = {
    math.round(predict(inputVector))
  }

  def predict(inputVector:  INDArray): Double = {
    Transforms.sigmoid(inputVector dot weights).sumT[Double]
  }


}
