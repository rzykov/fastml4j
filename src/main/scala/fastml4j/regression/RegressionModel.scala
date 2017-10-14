package fastml4j.regression

import fastml4j.util.Intercept
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

abstract class RegressionModel extends Intercept {

  var weights: INDArray = Nd4j.zeros(1)
  var interceptValue: Float = 0
  var losses: Seq[Float] = Seq[Float]()

  def fit(dataSet: DataSet, initWeights: Option[INDArray]): Unit

  def predict(inputVector: INDArray): Float

}
