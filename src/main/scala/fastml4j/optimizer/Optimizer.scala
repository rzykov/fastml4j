package fastml4j.optimizer


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.loss.Loss
import org.nd4j.linalg.dataset.DataSet

/**
  * Abstrat class for optimizers (gradient descent, SGD etc)
  *
  * Created by rzykov on 23/06/17.
  */
abstract class Optimizer {
  def optimize(loss: Loss, initWeights: INDArray, dataSet: DataSet): (INDArray, Seq[Float])
}
