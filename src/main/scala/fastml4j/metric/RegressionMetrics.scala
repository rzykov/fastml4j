package fastml4j.metric

import fastml4j.classification.ClassificationModel
import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j



class RegressionMetrics(val real: INDArray, val predictedLabels: INDArray) {


  lazy val rootDifferenceSumSquared: Float =  Nd4j.norm2(real - predictedLabels).sumFloat

  lazy val differenceSumSquared: Float =  rootDifferenceSumSquared * rootDifferenceSumSquared

  lazy val differenceVariance: Float = (real - predictedLabels).variance

  lazy val numCases: Float = real.rows().toFloat

  require(numCases > 1, "use matrix rather than vectors")

  def meanSquaredError: Float = differenceSumSquared / numCases

  def rootMeanSquaredError: Float = rootDifferenceSumSquared / math.sqrt(numCases).toFloat

  lazy val meanAbsoluteError: Float = Nd4j.norm1(real - predictedLabels).sumFloat

  override def toString = s"rootMeanSquaredError: $rootMeanSquaredError  \nmeanSquaredError $meanSquaredError \nmeanAbsoluteerror $meanAbsoluteError\n"

}
