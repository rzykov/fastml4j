package fastml4j.loss
import org.nd4j.linalg.api.ndarray.INDArray
import fastml4j.util.Implicits._
//

sealed trait Regularisation {
  def lossRegularisation(weights: INDArray): Float
  def gradientRegularisation(weights: INDArray): INDArray
}

object NoRegularisation extends Regularisation {
  override def lossRegularisation(weights: INDArray): Float = 0
  override def gradientRegularisation(weights: INDArray): INDArray = weights * 0
}

class L2(protected val lambdaL2: Float) extends Regularisation {
  override def lossRegularisation(weights: INDArray): Float = (weights * weights).sumFloat * lambdaL2 / 2
  override def gradientRegularisation(weights: INDArray): INDArray = weights*lambdaL2
}

object L2{
  def apply(lambdaL2: Float) = new L2(lambdaL2)
}

