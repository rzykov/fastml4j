package mlscala.classification

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by rzykov on 23/06/17.
  */
abstract class ClassificationModel {

  def fit(trainData: INDArray, labels: INDArray)

  def predict(inputVector: INDArray): Double

  def predictRaw(inputVector: INDArray): Double

}
