package com.deepwissen.ml.utils

/**
 * Created by hendri_k on 6/25/15.
 */
trait Denomination[T]{
  def get : T
}


case class FieldValue(n : Double) extends Denomination[Double]{
  override def get = n
}

case class TargetValue(n : List[Double]) extends Denomination[List[Double]]{
  override def get = n
}
//
//object FieldValue {
//  def apply(x : Double): FieldValue = {
//    new FieldValue(x)
//  }
//}
//
//object TargetValue {
//  def apply(x : List[Double]): TargetValue = {
//    new TargetValue(x)
//  }
//}