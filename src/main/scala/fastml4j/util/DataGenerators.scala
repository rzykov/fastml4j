package fastml4j.util

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import fastml4j.util.Implicits._

import scala.util.Random

object DataGenerators {

  /**
    * @param intercept Data intercept
    * @param weights  Weights to be applied.
    * @param xMean the mean of the generated features. Lots of time, if the features are not properly
    *              standardized, the algorithm with poor implementation will have difficulty
    *              to converge.
    * @param xVariance the variance of the generated features.
    * @param nPoints Number of points in sample.
    * @param seed Random seed
    * @param eps Epsilon scaling factor.
    * @return Seq of input.
    */
  def generateLinearInput(
    intercept: Float,
    weights: Array[Float],
    xMean: Array[Float],
    xVariance: Array[Float],
    nPoints: Int,
    seed: Int,
    eps: Float): DataSet = {

    val rnd = new Random(seed)
    def rndElement(i: Int) = {(rnd.nextFloat() - 0.5f) * math.sqrt(12.0f * xVariance(i)) + xMean(i)}

    val seq = (0 until nPoints).map { _ =>
      val features = weights.indices.map ( rndElement ).toArray.toNDArray
      val label = (weights.toNDArray dot features.T).sumFloat + intercept + eps * rnd.nextGaussian()
      (label, features) }
      .toArray

    val features = Nd4j.vstack(seq.map(_._2):_*)
    val labels = seq.map(_._1).map(Array(_)).toNDArray

    new DataSet(features, labels)
  }

  //from Spark test
  def generateLogisticInput(
    offset: Float,
    weights: Array[Float],
    nPoints: Int,
    seed: Int): (Array[Array[Float]], Array[Array[Float]]) = {
    val rnd = new util.Random(seed)
    val features = Array.fill[Array[Float]](nPoints)(weights.map(_ => rnd.nextGaussian().toFloat))
    val xWeighted = features.map{ featuresArr => featuresArr.zip(weights).map{case(x,y) => x*y}.sum}

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + xWeighted(i))))
      if (rnd.nextFloat() < p) Array(1.0f) else Array(0.0f)}
      .toArray

    (features, y)
  }

}
