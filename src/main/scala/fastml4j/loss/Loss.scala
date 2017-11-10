
package fastml4j.loss


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.Implicits._

abstract class Loss{

  /** Calculates the loss function value
    *
    * @param weights input weights vector
    * @param dataSet training dataset
    * @return computed value of the loss function
    */
  def loss(weights: INDArray, dataSet: DataSet): Float


  /** Calculates the gradient of loss function value
    *
    * @param weights input weights vector
    * @param dataSet input weights vector
    * @return computed value of the gradient of the loss function
    */
  def gradient(weights: INDArray,dataSet: DataSet): INDArray


  /** Calculate the gradient numerically, useful for a testing of gradient functions
    * More about gradient checking:  http://cs231n.github.io/neural-networks-3/
    * @param weights input weights vector
    * @param dataSet input weights vector
    * @param eps delta parameter for a derivative calculation
    * @return computed value of the gradient of the loss function
    */
  def numericGradient(weights: INDArray, dataSet: DataSet, eps: Float = 1e-6f): INDArray =
    (0 until weights.columns).map {
      i =>
        val oldWeights = weights.dup.put(0, i, weights.get(0,i) - eps)
        val newWeights = weights.dup.put(0, i, weights.get(0,i) + eps)

        (loss(newWeights, dataSet) - loss(oldWeights, dataSet)) / 2.0f / eps }
      .toArray
      .toNDArray

}
