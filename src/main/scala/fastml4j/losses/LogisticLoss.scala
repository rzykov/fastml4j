package fastml4j.losses

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * Created by rzykov on 31/05/17.
  */


class LogisticLoss(lambdaL2: Double) extends Loss {

  private def sigmoid(weights: INDArray, data: INDArray): INDArray = Transforms.sigmoid(data dot weights.T)

  //J = 1/m*sum(dot(-y,log(sigmoid(X*theta)))-dot(1-y,log(1-sigmoid(X*theta))));
  def loss(weights: INDArray, trainData: INDArray, labels: INDArray): Double = {

    val sigmoidVec = sigmoid(weights, trainData)
    val lossVec = ((labels.neg) dot (Transforms.log(sigmoidVec))) -
      ((labels.neg + 1.0) dot Transforms.log(((sigmoidVec.neg + 1.0))))
    val regularized: Double = (weights * weights).sumT[Double] * lambdaL2 / 2

    (lossVec.sumT[Double] / trainData.rows) + regularized
  }

  //grad = 1/m*sum((sigmoid(X*theta)-y).*X,1)';
  def gradient(weights: INDArray, trainData: INDArray, labels: INDArray): INDArray = {
    val main = ((trainData.T) dot (sigmoid(weights, trainData) - labels.T))
    val regularized = weights * lambdaL2

    (main.T) / (trainData.rows) + regularized
  }

}

