package mlscala.optimizer

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import mlscala.losses.Loss

/**
  * Created by rzykov on 23/06/17.
  */
abstract class Optimizer {

  def optimize(loss: Loss, initWeights: INDArray, trainData: INDArray, labels: INDArray): (INDArray, Seq[Double])

}
