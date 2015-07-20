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
trait SynapsysFactory[T] {

  /**
   * Create new synapsys from perceptron to perceptron
   * @param from from perceptron
   * @param to to perceptron
   * @return synapsys
   */
  def apply(from: Perceptron, to: Perceptron): Synapsys
  def apply(listOfSynapsys: List[Synapsys]): List[Synapsys]
  def getSynapsys() : T

}

/**
 * Factory object for create synapsys with random weight
 * @param value this value will multiplied with Math.random() value
 *              for generate new synapsys weight
 * @author Eko Khannedy
 */
case class RandomSynapsysFactory(value: Double = 0.05) extends SynapsysFactory[Synapsys] {

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
  def apply(listOfSynapsys: List[Synapsys]): List[Synapsys] = null

//  override def apply(listOfSynapsys: List[Synapsys]): List[Synapsys]

  override def getSynapsys() : Synapsys = null
}


case class CopySynapsysFactory(listOfSynapsyses: List[Synapsys]) extends SynapsysFactory[List[Synapsys]] {

//  override def apply(from: Perceptron, to: Perceptron): Synapsys
  /**
   * Create new synapsys from perceptron to perceptron with random synapsys
   * @param listOfSynapsys from another network
   * @return synapsys
   */
  override def apply(listOfSynapsys: List[Synapsys]): List[Synapsys] = {
    this(listOfSynapsys)
    listOfSynapsys
  }

  override def apply(from: Perceptron, to: Perceptron): Synapsys = null

  override def getSynapsys() : List[Synapsys] = listOfSynapsyses
}