package fastml4j.optimizer

import fastml4j.losses.Loss
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

import scala.annotation.tailrec

/**
  * Created by rzykov on 23/06/17.
  */
/*class GradientDescentDecreasingLearningRate(
  val maxIterations: Int,
  val alpha: Double,
  val eps: Double = 1e-6) extends Optimizer {


  override def optimize(loss: Loss, initWeights: INDArray, dataSet: DataSet, dataSet.getLabels: INDArray)
    : (INDArray, Seq[Double]) = {

    @tailrec
    def helperOptimizer( prevWeights:INDArray, losses: Seq[Double]): (INDArray, Seq[Double]) = {
      val step = losses.size
      println(step)
      val gradient = loss.gradient(prevWeights, dataSet.getFeatures, dataSet.getLabels)
      val decreasingStepRate = 1.0 / (step + 1.0)/ gradient.norm2Number().doubleValue()
      println("decreasing " + gradient * alpha * decreasingStepRate)
      val weights = prevWeights - gradient * alpha * decreasingStepRate
      val currentLoss = loss.loss(weights, dataSet.getFeatures, dataSet.getLabels)
      println(currentLoss)
      println(gradient + " " + weights)

      if( step > 0 && ((math.abs(currentLoss - losses.last) < eps) || step >= maxIterations))
        (weights, losses :+ currentLoss)
      else
        helperOptimizer(weights, losses :+ currentLoss)}

    helperOptimizer(initWeights, Seq[Double]())
  }

}*/
