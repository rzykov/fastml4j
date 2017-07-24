
package fastml4j.losses

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j


abstract class Loss{
  protected var lambdaL2: Double = 0
  protected var lambdaL1: Double = 0

  def loss(weights: INDArray, dataSet: DataSet): Double
  def gradient(weights: INDArray,dataSet: DataSet): INDArray

  // More about gradient checking:  http://cs231n.github.io/neural-networks-3/
  def numericGradient(weights: INDArray, dataSet: DataSet, eps: Double = 1e-6): INDArray =
    (0 to (weights.columns() - 1)).map {
      i =>
        val oldWeights = weights.dup.put(0, i, weights.get(0,i) - eps)
        val newWeights = weights.dup.put(0, i, weights.get(0,i) + eps)

        (loss(newWeights, dataSet) - loss(oldWeights, dataSet)) / 2.0 / eps }
      .toNDArray

}
