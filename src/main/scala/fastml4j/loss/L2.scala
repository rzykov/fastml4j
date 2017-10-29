package fastml4j.loss

import org.nd4j.linalg.api.ndarray.INDArray
import fastml4j.util.Implicits._
import fastml4j.util.Intercept
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex


trait L2  extends Loss {
  self: Intercept => //self type composition

  val lambdaL2: Float

  def weightsWithoutIntercept(weights: INDArray): INDArray =
    if (calcIntercept) {
      val newWeights = weights.dup()
      newWeights.putScalar(0,  weights.size(1) - 1, 0f) }
    else weights

  // aspect oriented programming
  abstract override def loss(weights: INDArray, dataSet: DataSet): Float = {
    val newWeights: INDArray = weightsWithoutIntercept(weights)
    super.loss(weights, dataSet) + (newWeights * newWeights).sumFloat * lambdaL2 / 2
  }

  abstract override def gradient(weights: INDArray, dataSet: DataSet): INDArray =
    super.gradient(weights, dataSet) + weightsWithoutIntercept(weights) * lambdaL2
}