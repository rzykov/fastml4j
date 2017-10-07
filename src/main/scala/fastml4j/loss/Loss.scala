
package fastml4j.loss


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.Implicits._

abstract class Loss{

  def regularisationFactor: Float = 0

  def loss(weights: INDArray, dataSet: DataSet): Float
  def gradient(weights: INDArray,dataSet: DataSet): INDArray

  // More about gradient checking:  http://cs231n.github.io/neural-networks-3/
  def numericGradient(weights: INDArray, dataSet: DataSet, eps: Float = 1e-6f): INDArray =
    (0 until weights.columns).map {
      i =>
        val oldWeights = weights.dup.put(0, i, weights.get(0,i) - eps)
        val newWeights = weights.dup.put(0, i, weights.get(0,i) + eps)

        (loss(newWeights, dataSet) - loss(oldWeights, dataSet)) / 2.0f / eps }
      .toArray
      .toNDArray

}
