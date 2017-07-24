package fastml4j.regression

import fastml4j.losses.OLSLoss
import fastml4j.optimizer.GradientDescent
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * Created by rzykov on 02/07/17.
  */
class LinearRegression(val lambdaL2: Double,
  val alpha: Double = 0.01,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Double = 1e-6) {

  var weights: INDArray = Nd4j.zeros(1)
  var losses: Seq[Double] = Seq[Double]()

  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(new OLSLoss(lambdaL2),
      initWeights.getOrElse(Nd4j.zeros(dataSet.numExamples)),
      dataSet)
    weights = weightsOut
    losses = lossesOut
  }

  def predict(inputVector: INDArray): Double = {
    (inputVector dot weights).sumT[Double]
  }


}
