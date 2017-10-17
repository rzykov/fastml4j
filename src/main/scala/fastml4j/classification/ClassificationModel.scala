package fastml4j.classification

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import fastml4j.util.Implicits._
import fastml4j.util.Intercept

/**
  * Created by rzykov on 23/06/17.
  */
abstract class ClassificationModel{
  var weights: INDArray = Nd4j.zeros(1)
  var losses: Seq[Float] = Seq[Float]()

  def fit(dataSet: DataSet, initWeights: Option[INDArray]): Unit

  def predictClass(inputVector: INDArray): Float

  def predict(inputVector: INDArray): Float

}

