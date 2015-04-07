/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.function

/**
 * @author Eko Khannedy
 * @since 3/1/15
 */
trait ThresholdFunction {

  /**
   * Compare result value1 to expected value2
   * @param value1 result value
   * @param value2 excepted value
   * @return true if ok, false if not ok
   */
  def compare(value1: Double, value2: Double): Boolean

}

/**
 * Simple implementation of Threshold Function
 */
object SimpleThresholdFunction extends ThresholdFunction {

  /**
   * Compare result value1 to expected value2
   * @param value1 result value
   * @param value2 excepted value
   * @return true if ok, false if not ok
   */
  override def compare(value1: Double, value2: Double): Boolean = value1 == value2
}

/**
 * Fix threshold function
 * @param value value
 * @param left left
 * @param right right
 */
case class EitherThresholdFunction(value: Double, left: Double, right: Double) extends ThresholdFunction {

  /**
   * Compare result value1 to expected value2
   * @param value1 result value
   * @param value2 excepted value
   * @return true if ok, false if not ok
   */
  override def compare(value1: Double, value2: Double): Boolean = {
    val current = if (value1 >= value) right else left
    current == value2
  }
}

/**
 * Threshold function with range value
 * @param range range
 */
case class RangeThresholdFunction(range: Double) extends ThresholdFunction {

  /**
   * Compare result value1 to expected value2
   * @param value1 result value
   * @param value2 excepted value
   * @return true if ok, false if not ok
   */
  override def compare(value1: Double, value2: Double): Boolean = {
    val min = value1 - range
    val max = value1 + range
    value2 >= min && value2 <= max
  }
}
