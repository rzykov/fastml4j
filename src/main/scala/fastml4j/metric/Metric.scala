package fastml4j.metric

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

//Classification:
// Recall
// F1
// Missclassifiaction matrix
// Precision
// AUC

// Regression
// Squared loss
// R2 share

// Ranking
// NDCG
// MAR??

abstract class Metric {

  def eval(labels: INDArray, labelsPredicted: INDArray)

}
