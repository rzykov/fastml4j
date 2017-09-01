package fastml4j.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

object Implicits  {


  implicit class ArrayFromIndarray(underlying: INDArray) {
    def toArrayDouble: Array[Array[Double]] =
      (0 until underlying.rows()).map { rowi => (0 until underlying.columns()).map { coli => underlying.getDouble(rowi, coli) }.toArray }
        .toArray

    def toArray: Array[Array[Double]] = toArrayDouble

    def toArrayFloat: Array[Array[Float]] =
      (0 until underlying.rows()).map { rowi => (0 until underlying.columns()).map { coli => underlying.getFloat(rowi, coli) }.toArray }
        .toArray

    def toArrayInt: Array[Array[Int]] =
      (0 until underlying.rows()).map { rowi => (0 until underlying.columns()).map { coli => underlying.getInt(rowi, coli) }.toArray }
        .toArray

  }

 /* implicit class DataSetImplicits(dataset: DataSet) {


  }*/

  implicit class IntWithIndarray(value: Int)  {
    def +(indArray: INDArray): INDArray = indArray + value
    def -(indArray: INDArray): INDArray = indArray - value
    def *(indArray: INDArray): INDArray = indArray * value
  }


  implicit class FloatWithIndarray(value: Float) {
    def +(indArray: INDArray): INDArray = indArray + value
    def -(indArray: INDArray): INDArray = indArray - value
    def *(indArray: INDArray): INDArray = indArray * value
  }

  implicit class DoubleWithIndarray(value: Double) {
    def +(indArray: INDArray): INDArray = indArray + value
    def -(indArray: INDArray): INDArray = indArray - value
    def *(indArray: INDArray): INDArray = indArray * value
  }


}
