/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * Factory trait for create Synapsys class
 * @author Eko Khannedy
 * @since 6/3/15
 */
trait SynapsysFactory {

  /**
   * Create new synapsys from perceptron to perceptron
   * @param from from perceptron
   * @param to to perceptron
   * @return synapsys
   */
  def apply(from: Perceptron, to: Perceptron): Synapsys

}

/**
 * Factory object for create synapsys with random weight
 * @param value this value will multiplied with Math.random() value
 *              for generate new synapsys weight
 * @author Eko Khannedy
 */
case class RandomSynapsysFactory(value: Double = 0.05) extends SynapsysFactory {

  /**
   * Create new synapsys from perceptron to perceptron with random synapsys
   * @param from from perceptron
   * @param to to perceptron
   * @return synapsys
   */
  override def apply(from: Perceptron, to: Perceptron): Synapsys = Synapsys(
    from = from,
    to = to,
    weight = value * Math.random()
  )
}
