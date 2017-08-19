package fastml4j.util

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

object RichIndarray {

  implicit class ArrayFromIndarray(indArray: INDArray) {
    def toArray: Array[Array[Double]] =
      (0 until indArray.rows()).map{ rowi => (0 to indArray.columns()).map{ coli => indArray.getDouble(rowi, coli)}.toArray }
        .toArray
  }

}
