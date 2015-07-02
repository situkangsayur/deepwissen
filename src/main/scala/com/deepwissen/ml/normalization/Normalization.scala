/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.normalization

import com.deepwissen.ml.utils.ContValue

/**
 * @author Eko Khannedy
 * @since 3/1/15
 * @tparam T dataset
 */
trait Normalization[T] {

  /**
   * Normalize dataset
   * @param dataset dataset
   * @return normal dataset
   */
  def normalize(dataset: T, targetClass : Int): T

  /**
   * Normalize value
   * @param value value
   * @param maxValue max value
   * @param minValue min value
   * @return normal value
   */
  def normalize(value: Double, minValue: Double, maxValue: Double, targetClass : Int) =
    ContValue((value - minValue) / (maxValue - minValue))

  /**
   * Denormalize normal value
   * @param normalValue normal value
   * @param index index of value
   * @param dataset dataset
   * @return double
   */
  def denormalize(normalValue: Double, index: Int, dataset: T, targetClass: Int): Double

  /**
   * Denormalize value
   * @param normalValue normal value
   * @param minValue min value
   * @param maxValue max value
   * @return value
   */
  def denormalize(normalValue: Double, minValue: Double, maxValue: Double, targetClass: Int) =
    (normalValue * (maxValue - minValue)) + minValue

}