package fastml4j.regression

import fastml4j.losses.OLSLoss
import fastml4j.optimizer.GradientDescent
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import fastml4j.util.Implicits._


/**
  * Created by rzykov on 02/07/17.
  */
class LinearRegression(val lambdaL2: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Float = 1e-6f,
  val standardize: Boolean = true) extends RegressionModel {

  var weights: INDArray = Nd4j.zeros(1)
  var losses: Seq[Float] = Seq[Float]()
  //var standardDeviationFeatures: INDArray
 // var standardDeviationLabels: Float



  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) = optimizer.optimize(new OLSLoss(lambdaL2),
      initWeights.getOrElse(Nd4j.zeros(dataSet.numInputs)),
      dataSet)
    weights = weightsOut
    losses = lossesOut
  }

  def predict(inputVector: INDArray): Float = {
    (inputVector dot weights).sumT
  }

  def transform(dataSet: DataSet): INDArray = (weights dot dataSet.getFeatures.T)

}
