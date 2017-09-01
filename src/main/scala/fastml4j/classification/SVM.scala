package fastml4j.classification

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.optimizer._
import fastml4j.losses._
import org.nd4j.linalg.dataset.DataSet
/**
  * Created by rzykov on 23/06/17.
  */
class SVM(val lambdaL2: Double,
  val alpha: Double = 0.01,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "PegasosSGD",
  val eps: Double = 1e-6) extends ClassificationModel {

  var weights: INDArray = Nd4j.zeros(1)
  var losses: Seq[Double] = Seq[Double]()


  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {

    dataSet.validate()

    val optimizer: Optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
    //  case "GradientDescentDecreasingLearningRate" => new GradientDescentDecreasingLearningRate(maxIterations, alpha, eps)
      case "PegasosSGD" => new PegasosSGD(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(
      new HingeLoss(lambdaL2),
      initWeights = initWeights.getOrElse(Nd4j.zeros(dataSet.numInputs)),
      dataSet)

    weights = weightsOut
    losses = lossesOut
  }

  def predictClass(inputVector: INDArray): Double = {
    val sign = math.signum((inputVector dot weights.T).sumT[Double])
    if( sign != 0 ) sign else 1.0
  }

  def predict(inputVector:  INDArray): Double = {
    (inputVector dot weights).sumT[Double]
  }


}
