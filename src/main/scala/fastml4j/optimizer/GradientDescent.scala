package fastml4j.optimizer


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.loss.Loss
import org.nd4j.linalg.dataset.DataSet
import fastml4j.util.Implicits._

import scala.annotation.tailrec

/**
  * Classical gradient descent
  *
  * Created by rzykov on 23/06/17.
  */
class GradientDescent(
  val maxIterations: Int,
  val stepSize: Float,
  val eps: Float = 1e-6f) extends Optimizer {

  override def optimize(loss: Loss, initWeights: INDArray, dataset: DataSet): (INDArray, Seq[Float]) = {

    @tailrec
    def helperOptimizer( prevWeights:INDArray, losses: Seq[Float]): (INDArray, Seq[Float]) = {
      val weights = prevWeights - loss.gradient(prevWeights, dataset) * stepSize
      val currentLoss = loss.loss(weights, dataset)

      if( losses.nonEmpty && ((math.abs(currentLoss - losses.last) < eps) || losses.size >= maxIterations))
        (weights, losses :+ currentLoss)
      else
        helperOptimizer(weights, losses :+ currentLoss)}

    helperOptimizer(initWeights, Seq[Float]())
  }

}
