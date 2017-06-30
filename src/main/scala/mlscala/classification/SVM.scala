package mlscala.classification

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import mlscala.optimizer._
import mlscala.losses._
/**
  * Created by rzykov on 23/06/17.
  */
class SVM(val c: Double,
  val alpha: Double = 0.0001,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Double = 1e-6) extends ClassificationModel {

  var weights: INDArray = Nd4j.zeros(1)

  def fit(trainData: INDArray, labels: INDArray) = {

    val initWeights: INDArray = Nd4j.rand(trainData.columns - 1)

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, losses) = optimizer.optimize(new HingeLoss(c), initWeights, trainData, labels)
    weights = weightsOut
  }

  def predict(inputVector: INDArray): Double = {
    val sign = math.signum((inputVector dot weights.T).sumT[Double])
    if( sign != 0 ) sign else 1.0
  }

  def predictRaw(inputVector:  INDArray): Double = {
    (inputVector dot weights).sumT[Double]
  }


}
