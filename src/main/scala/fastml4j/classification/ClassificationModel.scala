package fastml4j.classification

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

/**
  * Created by rzykov on 23/06/17.
  */
abstract class ClassificationModel {

  def fit(dataSet: DataSet, initWeights: Option[INDArray]): Unit

  def predictClass(inputVector: INDArray): Double

  def predict(inputVector: INDArray): Double

}
