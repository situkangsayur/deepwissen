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

  /**
   * Activation function for t (weight)
   * @param weight t
   * @return function result
   */
  def activation(weight: Double): Double

  /**
   * Derivation function for t (weight)
   * @param weight t
   * @return function result
   */
  def derivation(weight: Double): Double

}

/**
 * A sigmoid function is a mathematical function having an "S" shape (sigmoid curve).
 * Often, sigmoid function refers to the special case of the logistic function shown
 * in the first figure and defined by the formula
 *
 * {{{
 *   S(t) = 1.0 / 1.0 + e^-t
 * }}}
 *
 * @see http://en.wikipedia.org/wiki/Sigmoid_function
 */
object SigmoidFunction extends ActivationFunction {

  override def activation(weight: Double): Double = 1.0 / (1 + Math.exp(-1.0 * weight))

  override def derivation(weight: Double): Double = weight * (1.0 - weight)
}