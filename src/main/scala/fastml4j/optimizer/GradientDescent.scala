package fastml4j.optimizer

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import fastml4j.losses.Loss

import scala.annotation.tailrec

/**
  * Created by rzykov on 23/06/17.
  */
class GradientDescent(
  val maxIterations: Int,
  val alpha: Double,
  val eps: Double = 1e-6) extends Optimizer {


  override def optimize(loss: Loss, initWeights: INDArray, trainData: INDArray, labels: INDArray)
    : (INDArray, Seq[Double]) = {

    @tailrec
    def helperOptimizer( prevWeights:INDArray, losses: Seq[Double]): (INDArray, Seq[Double]) = {
      val weights = prevWeights - loss.gradient(prevWeights, trainData, labels) * alpha
      val currentLoss = loss.loss(weights, trainData, labels)

      if( losses.size > 0 && ((math.abs(currentLoss - losses.last) < eps) || losses.size >= maxIterations))
        (weights, losses :+ currentLoss)
      else
        helperOptimizer(weights, losses :+ currentLoss)}

    helperOptimizer(initWeights, Seq[Double]())
  }

}
