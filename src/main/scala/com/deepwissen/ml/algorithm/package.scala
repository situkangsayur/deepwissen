package com.deepwissen.ml

import java.util.UUID

/**
 * @author Eko Khannedy
 * @since 4/7/15
 */
package object algorithm {

  /**
   * Generate random layer id
   * @return string
   */
  def newLayerId(): String = s"layer_${uuid()}"

  /**
   * Generate random bias id
   * @return string
   */
  def newBiasId(): String = s"bias_${uuid()}"

  /**
   * Generate random perceptron id
   * @return string
   */
  def newPerceptronId(): String = s"perceptron_${uuid()}"

  /**
   * Generate random uuid without - (minus, strip) character
   * @return uuid without - (minus, strip) char
   */
  def uuid(): String = UUID.randomUUID().toString.replaceAll("-", "")

  /**
   * Generate random weight
   * @return double
   */
  def newSynapsysWeight(): Double = 0.05 * Math.random()

}
