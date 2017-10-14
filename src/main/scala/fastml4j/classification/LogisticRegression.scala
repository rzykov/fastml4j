package fastml4j.classification

/**
  * Created by rzykov on 13/07/17.
  */
import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.optimizer._
import fastml4j.loss._
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms


/**
  * Created by rzykov on 13/07/17.
  */


class LogisticRegression
  (val regularisationFactor: Float,
  val alpha: Float = 0.01f,
  val maxIterations: Int = 1000,
  val stohasticBatchSize: Int = 100,
  val optimizerType: String = "GradientDescent",
  val eps: Float = 1e-6f,
  val intercept: Boolean = true) extends ClassificationModel {


  def fit(dataSet: DataSet, initWeights: Option[INDArray] = None): Unit = {
    dataSet.validate()

    val dataSetIntercept: DataSet = dataSetWithIntercept(dataSet, intercept)

    val optimizer = optimizerType match {
      case "GradientDescent" => new GradientDescent(maxIterations, alpha, eps)
      case _ => throw new Exception("Optimizer %s is not supported".format(optimizerType))
    }

    val (weightsOut, lossesOut) =
      optimizer.optimize(
        new LogisticLoss(L2(regularisationFactor, intercept)),
        initWeights = initWeights.getOrElse(Nd4j.zeros(dataSetIntercept.numInputs)),
        dataset = dataSetIntercept)

    interceptValue = extractIntercept(weightsOut, intercept)
    weights = extractWeights(weightsOut, intercept)
    losses = lossesOut
  }

  def predictClass(inputVector: INDArray): Float = {
    math.round(predict(inputVector))
  }

  def predict(inputVector:  INDArray): Float = {
    Transforms.sigmoid(inputVector dot weights + interceptValue).sumFloat
  }

}
