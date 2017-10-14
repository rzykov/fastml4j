package fastml4j.loss
import org.nd4j.linalg.api.ndarray.INDArray
import fastml4j.util.Implicits._
import org.nd4j.linalg.indexing.NDArrayIndex
//

sealed trait Regularisation {
  def hasIntercept: Boolean
  def lossRegularisation(weights: INDArray): Float
  def gradientRegularisation(weights: INDArray): INDArray
}

object NoRegularisation extends Regularisation {
  override def hasIntercept: Boolean = false
  override def lossRegularisation(weights: INDArray): Float = 0
  override def gradientRegularisation(weights: INDArray): INDArray = weights * 0
}

class L2 protected(val lambdaL2: Float, val hasIntercept: Boolean) extends Regularisation {

  def weightsWithoutIntercept(weights: INDArray): INDArray =
     if(hasIntercept) weights.get(NDArrayIndex.all, NDArrayIndex.interval(0, weights.size(1) - 1)) else weights

  override def lossRegularisation(weights: INDArray): Float = {
    val newWeights: INDArray = weightsWithoutIntercept(weights)
    (newWeights * newWeights).sumFloat * lambdaL2 / 2
  }

  override def gradientRegularisation(weights: INDArray): INDArray = weightsWithoutIntercept(weights)*lambdaL2
}

object L2{
  def apply(lambdaL2: Float, hasIntercept: Boolean = false) = new L2(lambdaL2, hasIntercept)
}

