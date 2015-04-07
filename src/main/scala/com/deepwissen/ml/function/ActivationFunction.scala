/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.function

/**
 * @author Hendri Karisma
 * @author Eko Khannedy
 * @since 11/15/14
 */
trait ActivationFunction extends Serializable {

  def activation(weight: Double): Double

  def derivation(weight: Double): Double

}

/**
 * Sigmoid Function
 */
object SigmoidFunction extends ActivationFunction {

  override def activation(weight: Double): Double = 1.0 / (1 + Math.exp(-1.0 * weight))

  override def derivation(weight: Double): Double = weight * (1.0 - weight)
}