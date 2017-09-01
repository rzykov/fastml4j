package fastml4j.regression

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

abstract class RegressionModel {

  def fit(dataSet: DataSet, initWeights: Option[INDArray]): Unit

  def predict(inputVector: INDArray): Double


}
