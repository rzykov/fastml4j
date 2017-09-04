package fastml4j.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4s.FloatNDArrayEvidence


object Implicits  {


  implicit val float = FloatNDArrayEvidence  //to use float default representation

  implicit class ArrayFromIndarray(underlying: INDArray) {
    def toArrayFloat: Array[Array[Float]] =
      (0 until underlying.rows()).map { rowi => (0 until underlying.columns()).map { coli => underlying.getFloat(rowi, coli) }.toArray }
        .toArray

    def toArray: Array[Array[Float]] = toArrayFloat

    def toArrayInt: Array[Array[Int]] =
      (0 until underlying.rows()).map { rowi => (0 until underlying.columns()).map { coli => underlying.getInt(rowi, coli) }.toArray }
        .toArray

  }


  implicit class IntWithIndarray(value: Int)  {
    def +(indArray: INDArray): INDArray = indArray + value
    def -(indArray: INDArray): INDArray = indArray - value
    def *(indArray: INDArray): INDArray = indArray * value
  }

  implicit class LongWithIndarray(value: Int)  {
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
