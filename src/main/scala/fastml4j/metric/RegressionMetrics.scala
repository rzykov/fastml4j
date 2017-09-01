package fastml4j.metric

import fastml4j.classification.ClassificationModel
import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._


class RegressionMetrics(val real: INDArray, val predictedLabels: INDArray) {


  lazy val rootDifferenceSumSquared: Double =  Nd4j.norm2(real - predictedLabels).sumT[Double]

  lazy val differenceSumSquared: Double =  rootDifferenceSumSquared * rootDifferenceSumSquared

  lazy val differenceVariance: Double = (real - predictedLabels).varT[Double]

  lazy val numCases: Double = real.rows().toDouble

  require(numCases > 1, "use matrix rather than vectors")

  def meanSquaredError: Double = differenceSumSquared / numCases

  def rootMeanSquaredError: Double = rootDifferenceSumSquared / math.sqrt(numCases)

  lazy val meanAbsoluteError: Double = Nd4j.norm1(real - predictedLabels).sumT[Double]

  override def toString = s"rootMeanSquaredError: $rootMeanSquaredError  \nmeanSquaredError $meanSquaredError \nmeanAbsoluteerror $meanAbsoluteError\n"

}
