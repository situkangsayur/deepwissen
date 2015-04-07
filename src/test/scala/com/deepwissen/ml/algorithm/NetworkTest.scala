/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import org.scalatest.FunSuite

/**
 * @author Eko Khannedy
 * @since 2/25/15
 */
class NetworkTest extends FunSuite {

  test("create network object") {
    val network = Network(5, 2)

    network.inputLayer.perceptrons.foreach { perceptron =>
      println(network.inputLayer.id + " input layer => " + perceptron.id)
    }
    println(network.inputLayer.id + " bias => " + network.inputLayer.bias.get.id)

    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        println(layer.id + " hidden layer => " + perceptron.id)
      }
      println(layer.id + " bias => " + layer.bias.get.id)
    }

    network.outputLayer.perceptrons.foreach { perceptron =>
      println(network.outputLayer.id + " output layer => " + perceptron.id)
    }

    network.synapsies.foreach { synapsys =>
      println(s"synapsys from ${synapsys.from} to ${synapsys.to} with weight ${synapsys.weight}")
    }

    println("total synapsys => " + network.synapsies.length)
  }

}
