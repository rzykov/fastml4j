package fastml4j.losses

import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

/**
  * Created by rzykov on 31/05/17.
  */


class LogisticLoss(lambdaL2: Double) extends Loss {

  private def sigmoid(weights: INDArray, data: INDArray): INDArray = Transforms.sigmoid(data dot weights.T)

  //J = 1/m*sum(dot(-y,log(sigmoid(X*theta)))-dot(1-y,log(1-sigmoid(X*theta))));
  def loss(weights: INDArray, dataSet: DataSet): Double = {

    val sigmoidVec = sigmoid(weights, dataSet.getFeatureMatrix)
    val lossVec = ((dataSet.getLabels.T.neg) dot (Transforms.log(sigmoidVec))) -
      ((1.0 + dataSet.getLabels.T.neg) dot Transforms.log((1.0 + sigmoidVec.neg)))
    val regularized: Double = (weights * weights).sumT[Double] * lambdaL2 / 2

    (lossVec.sumT[Double] / dataSet.numExamples) + regularized
  }

  //grad = 1/m*sum((sigmoid(X*theta)-y).*X,1)';
  def gradient(weights: INDArray, dataSet: DataSet): INDArray = {
    val main = ((dataSet.getFeatures.T) dot (sigmoid(weights, dataSet.getFeatures) - dataSet.getLabels))
    val regularized = weights * lambdaL2

    (main.T) / (dataSet.numExamples) + regularized
  }

}

