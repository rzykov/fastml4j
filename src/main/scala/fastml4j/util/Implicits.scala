package fastml4j.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

object Implicits  {

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
    def +(indArray: INDArray): INDArray = indArray.add(value)
    def -(indArray: INDArray): INDArray = indArray.sub(value)
    def *(indArray: INDArray): INDArray = indArray.mul(value)
  }

  implicit class LongWithIndarray(value: Long)  {
    def +(indArray: INDArray): INDArray = indArray.add(value)
    def -(indArray: INDArray): INDArray = indArray.sub(value)
    def *(indArray: INDArray): INDArray = indArray.mul(value)
  }

  implicit class FloatWithIndarray(value: Float) {
    def +(indArray: INDArray): INDArray = indArray.add(value)
    def -(indArray: INDArray): INDArray = indArray.sub(value)
    def *(indArray: INDArray): INDArray = indArray.mul(value)
  }

  implicit class DoubleWithIndarray(value: Double) {
    def +(indArray: INDArray): INDArray = indArray.add(value)
    def -(indArray: INDArray): INDArray = indArray.sub(value)
    def *(indArray: INDArray): INDArray = indArray.mul(value)
  }

  implicit class INDArrayWithINDArray(value: INDArray)  {
    def +(indArray: INDArray): INDArray = value.add(indArray)
    def -(indArray: INDArray): INDArray = value.sub(indArray)
    def *(indArray: INDArray): INDArray = value.mul(indArray)
    def dot(indArray: INDArray): INDArray = value.mmul(indArray)
    def T: INDArray = value.transpose()
    def sumFloat(): Float = value.sumNumber().floatValue()
    def unary_-(): INDArray = value.neg()
    def get(i: Int, j: Int): Float = value.getFloat(i, j)
    def variance: Float = value.varNumber().floatValue()
  }

  implicit class INDArrayWithFloat(indArray: INDArray)  {
    def +(value: Float): INDArray = indArray.add(value)
    def -(value: Float): INDArray = indArray.sub(value)
    def *(value: Float): INDArray = indArray.mul(value)
    def /(value: Float): INDArray = indArray.div(value)

  }

  implicit class FloatArray2INDArray(val array: Array[Float]){
    def toNDArray: INDArray = Nd4j.create(array)
  }

  implicit class FloatFloatArray2INDArray(val array: Array[Array[Float]]){
    def toNDArray: INDArray = Nd4j.create(array)
  }

  implicit class DoubleArray2INDArray(val array: Array[Double]){
    def toNDArray: INDArray = Nd4j.create(array)
  }

  implicit class DoubleDoubleArray2INDArray(val array: Array[Array[Double]]) {
    def toNDArray: INDArray = Nd4j.create(array)
  }

  implicit class IntArray2INDArray(val array: Array[Int]) {
    def toNDArray: INDArray = Nd4j.create(array.map(_.toFloat))
  }

  implicit class IntIntArray2INDArray(val array: Array[Array[Int]])  {
    def toNDArray: INDArray = Nd4j.create(array.map(_.map(_.toFloat)))
  }


}
