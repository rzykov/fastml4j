package fastml4j.optimizer

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.losses.Loss
import org.nd4j.linalg.dataset.DataSet

/**
  * Created by rzykov on 23/06/17.
  */
abstract class Optimizer {

  def optimize(loss: Loss, initWeights: INDArray, dataSet: DataSet): (INDArray, Seq[Float])

}
