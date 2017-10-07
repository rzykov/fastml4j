package fastml4j.optimizer

import fastml4j.loss.Loss
import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j


import scala.annotation.tailrec

/**
  * Created by rzykov on 18/07/17.
  */
class PegasosSGD(
  val maxIterations: Int,
  val lambda: Float,
  val eps: Float = 1e-6f,
  val batchSize: Int = 100,
  val withReplacement: Boolean = true) extends Optimizer {

  //http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
  def optimize(loss: Loss, initWeights: INDArray, dataSet: DataSet): (INDArray, Seq[Float]) = {

    @tailrec
    def helperOptimizer( prevWeights:INDArray, losses: Seq[Float], batch: Int): (INDArray, Seq[Float]) = {
      val sampleDataSet = dataSet.sample(batchSize, withReplacement)
      val eta = 1.0f / lambda / batch
      val weights = prevWeights * (1 - lambda * eta)  -
        loss.gradient(prevWeights, sampleDataSet) * eta

      val currentLoss = loss.loss(weights, sampleDataSet)

      if( losses.nonEmpty && ((math.abs(currentLoss - losses.last) < eps) || losses.size >= maxIterations))
        (weights, losses :+ currentLoss)
      else
        helperOptimizer(weights, losses :+ currentLoss, batch + 1)}

    helperOptimizer(initWeights, Seq[Float](), 1)

  }


}

