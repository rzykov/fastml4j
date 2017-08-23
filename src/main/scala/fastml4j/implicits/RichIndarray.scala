package fastml4j.implicits

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

object RichIndarray {

  implicit class ArrayFromIndarray(indArray: INDArray) {
    def toArrayDouble: Array[Array[Double]] =
      (0 until indArray.rows()).map { rowi => (0 until indArray.columns()).map { coli => indArray.getDouble(rowi, coli) }.toArray }
        .toArray

    def toArray: Array[Array[Double]] = toArrayDouble

    def toArrayFloat: Array[Array[Float]] =
      (0 until indArray.rows()).map { rowi => (0 until indArray.columns()).map { coli => indArray.getFloat(rowi, coli) }.toArray }
        .toArray

    def toArrayInt: Array[Array[Int]] =
      (0 until indArray.rows()).map { rowi => (0 until indArray.columns()).map { coli => indArray.getInt(rowi, coli) }.toArray }
        .toArray
  }
}
