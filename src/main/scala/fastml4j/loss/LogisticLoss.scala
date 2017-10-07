package fastml4j.loss

import fastml4j.util.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
//

/**
  * Created by rzykov on 31/05/17.
  */


class LogisticLoss[T <: Regularisation](regularisation: T = NoRegularisation) extends Loss {

  private def sigmoid(weights: INDArray, data: INDArray): INDArray = Transforms.sigmoid(data dot weights.T)

  //J = 1/m*sum(dot(-y,log(sigmoid(X*theta)))-dot(1-y,log(1-sigmoid(X*theta))));
  override def loss(weights: INDArray, dataSet: DataSet): Float = {

    val sigmoidVec = sigmoid(weights, dataSet.getFeatureMatrix)
    val lossVec = ((dataSet.getLabels.T.neg) dot (Transforms.log(sigmoidVec))) -
      ((1.0f + dataSet.getLabels.T.neg) dot Transforms.log((1.0f + sigmoidVec.neg)))

    (lossVec.sumFloat / dataSet.numExamples) + regularisation.lossRegularisation(weights)
  }

  //grad = 1/m*sum((sigmoid(X*theta)-y).*X,1)';
  override def gradient(weights: INDArray, dataSet: DataSet): INDArray = {
    val main = ((dataSet.getFeatures.T) dot (sigmoid(weights, dataSet.getFeatures) - dataSet.getLabels))

    (main.T) / (dataSet.numExamples) + regularisation.lossRegularisation(weights)
  }

}

