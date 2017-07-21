package fastml4j.optimizer

import fastml4j.losses.Loss
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.annotation.tailrec

/**
  * Created by rzykov on 18/07/17.
  */
class PegasosSGD(
  val maxIterations: Int,
  val lambda: Double,
  val eps: Double = 1e-6,
  val batchSize: Int = 100,
  val withReplacement: Boolean = true) extends Optimizer {

  def sampleWithReplacementByRow(data: INDArray, labels: INDArray, sampleSize: Int ): (INDArray, INDArray) = {
    val index = (0 until data.rows).toArray.toNDArray
    val probs = (0 until data.rows).map(_ => 1.0/data.rows).toArray.toNDArray
    val sample = Nd4j.choice(index, probs, sampleSize)
    val sampleIndex =  for( i <- 0 until sampleSize) yield {sample.getInt(i)}
    (data.getRows(sampleIndex.toArray : _*).dup(), labels.getColumns(sampleIndex.toArray : _*).dup())
  }

  //http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
  def optimize(loss: Loss, initWeights: INDArray, trainData: INDArray, labels: INDArray): (INDArray, Seq[Double]) = {

    @tailrec
    def helperOptimizer( prevWeights:INDArray, losses: Seq[Double], batch: Int): (INDArray, Seq[Double]) = {
      val (sampleData, sampleLabels) = sampleWithReplacementByRow(trainData, labels, batchSize)
      val eta = 1.0 / lambda / batch
      val weights = prevWeights * (1 - lambda * eta)  -
        loss.gradient(prevWeights, sampleData, sampleLabels) * eta

      val currentLoss = loss.loss(weights, sampleData, sampleLabels)

      if( losses.size > 0 && ((math.abs(currentLoss - losses.last) < eps) || losses.size >= maxIterations))
        (weights, losses :+ currentLoss)
      else
        helperOptimizer(weights, losses :+ currentLoss, batch + 1)}

    helperOptimizer(initWeights, Seq[Double](), 1)

  }


}

