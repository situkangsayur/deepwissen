/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.function.ActivationFunction

/**
 * Layer model class
 * @author Eko Khannedy
 * @since 2/25/15
 */
trait Layer {

  /**
   * Get layer id
   * @return id
   */
  def id: String

  /**
   * Get list of layer perceptrons
   * @return list of perceptron
   */
  def perceptrons: List[Perceptron]

  /**
   * Get first perceptrons
   * @return perceptron
   */
  def perceptron: Perceptron = perceptrons.head

  /**
   * Get map of index and perceptron
   */
  val perceptronsWithIndex: Map[Int, Perceptron] = perceptrons.map(p => (p.index, p)).toMap

  /**
   * Get bias
   * @return some bias or none
   */
  def bias: Option[Perceptron]

  /**
   * Get next layer
   */
  var next: Option[Layer]

  /**
   * get previous layer
   */
  var prev: Option[Layer]

  /**
   * Fill perceptrons output with given dataset and update bias output to 1.0
   * @param data dataset
   * @return this layer
   */
  def fillOutput(data: Array[Any]): Layer = {
    val tempData = data.filterNot(p => p.isInstanceOf[List[Double]])
    // update all perceptrons with given dataset
    perceptrons.foreach { perceptron =>
      // make sure non error ArrayIndexOutOfBoundsException
      if (perceptron.index >= 0 && perceptron.index < tempData.length)
        perceptron.output = tempData(perceptron.index).asInstanceOf[Double]
    }
    // update bias to 1.0
    bias.foreach(bias => bias.output = 1.0)
    this
  }

  /**
   * Fill perceptrons weight with given dataset and automatically update output
   * with activation function, and update bias output
   * @param data dataset
   * @param activationFunction activation function
   * @return this layer
   */
  def fillWeight(data: List[Any], activationFunction: ActivationFunction): Layer = {
    // update all perceptrons with given dataset
    perceptrons.foreach { perceptron =>
      perceptron.weight = data(perceptron.index).asInstanceOf[Double]
      perceptron.output = activationFunction.activation(perceptron.weight)
    }
    // update bias to 1.0
    bias.foreach(bias => bias.output = 1.0)
    this
  }

}

/**
 * Input Layer
 */
case class InputLayer(id: String,
                      perceptrons: List[Perceptron],
                      bias: Option[Perceptron],
                      var next: Option[Layer] = None) extends Layer {

  /**
   * Input layer doesn't have previous layer, it will allways return None
   */
  override var prev: Option[Layer] = None
}

/**
 * Hidden Layer
 */
case class HiddenLayer(id: String,
                       perceptrons: List[Perceptron],
                       bias: Option[Perceptron],
                       var prev: Option[Layer] = None,
                       var next: Option[Layer] = None) extends Layer

/**
 * Output Layer
 */
case class OutputLayer(id: String,
                       perceptrons: List[Perceptron],
                       var prev: Option[Layer] = None,
                       var next: Option[Layer] = None) extends Layer {

  /**
   * Output layer doesn't have bias, it will always return None
   */
  override val bias: Option[Perceptron] = None
}
