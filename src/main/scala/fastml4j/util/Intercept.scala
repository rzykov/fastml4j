package fastml4j.util
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex
import fastml4j.util.Implicits._


trait Intercept {

  val intercept: Boolean
  var interceptValue: Float = 0f

  def dataSetWithIntercept(dataSet: DataSet): DataSet =
    if (intercept) dataSet.addIntercept else dataSet

  def extractIntercept(weights: INDArray): Float =
    if (intercept) weights.getFloat(0, weights.size(1) - 1) else 0

  def extractWeights(weights: INDArray): INDArray =
    if (intercept) weights.get(NDArrayIndex.all, NDArrayIndex.interval(0, weights.size(1) - 1)) else weights
}
