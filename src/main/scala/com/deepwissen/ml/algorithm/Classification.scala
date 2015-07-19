/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.function.ActivationFunction
import com.deepwissen.ml.utils.{BinaryValue, Denomination}

import scala.concurrent.{ExecutionContext, Future}

/**
 * Base trait for classification
 * @author Eko Khannedy
 * @since 6/3/15
 */
trait Classification[DATA, MODEL] {

  /**
   * Run classification
   * @param data data
   * @param model model
   * @param activationFunction activation function
   * @return classification result
   */
  def apply(data: DATA, model: MODEL, activationFunction: ActivationFunction): Denomination[_]

  /**
   * Run classification async
   * @param data data
   * @param model model
   * @param activationFunction activation function
   * @param executionContext execution context
   * @return
   */
  def async(data: DATA, model: MODEL, activationFunction: ActivationFunction)
           (implicit executionContext: ExecutionContext): Future[Denomination[_]] =
    Future(apply(data, model, activationFunction))

}

/**
 * Basic implementation of classification for data array of double and network model
 * @author Eko Khannedy
 * @since 6/3/15
 */
object BasicClassification extends Classification[Array[Denomination[_]], Network] {

  /**
   * Run classification
   * @param data data
   * @param network model
   * @param activationFunction activation function
   * @return classification result
   */
  override def apply(data: Array[Denomination[_]], network: Network, activationFunction: ActivationFunction): Denomination[_] = {

    // fill input layer
    network.inputLayer.fillOutput(data)

    // fill hidden layer
    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightTo(perceptron)
        perceptron.output = activationFunction.activation(perceptron.weight)
      }
    }

    // fill output layer
    network.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = activationFunction.activation(perceptron.weight)
    }

    BinaryValue(network.outputLayer.perceptrons.map(x => x.output))
    // calculate result
//    network.outputLayer.perceptrons.foldLeft(0.0) { (value, perceptron) =>
//      value + perceptron.output
//    }
  }
}

object RBMClassification extends Classification[Array[Denomination[_]], MarkovChain]{

  override def apply(data: Array[Denomination[_]], network: MarkovChain, activationFunction: ActivationFunction): Denomination[_] = {


    var classifyNetwork = Network(
//      inputPerceptronSize = network.inputLayer.perceptrons.size,
//      hiddenSize = 1,
//      outputPerceptronSize = network.inputLayer.perceptrons.size,
//      synapsysFactory = CopySynapsysFactory(network.synapsies)
//    )


    val inputLayer = new InputLayer(
      id = newLayerId(),
      perceptrons = network.inputLayer.perceptrons.map( p => p),
      bias = Some(Network.newBias(network.inputLayer.bias.get.id))
    )

//    classifyNetwork.inputLayer = inpuLayer
    var prevLayer: Layer = inputLayer


    val hiddenLayer = new HiddenLayer(
      id = newLayerId(),
      perceptrons = network.hiddenLayer.perceptrons.map( p => p),
      bias = Some(Network.newBias(network.hiddenLayer.bias.get.id))
    )

    prevLayer.next = Some(hiddenLayer)
    hiddenLayer.prev = Some(prevLayer)

    prevLayer = hiddenLayer
    val outputLayer = new OutputLayer(
      id = newLayerId(),
      perceptrons = network.inputLayer.perceptrons.map( p => p),
      prev = Some(prevLayer)
    )
    prevLayer.next = Some(outputLayer)


    val newNetwork = new Network(inputLayer, hiddenLayers, outputLayer, synapsies)

    BinaryValue(classifyNetwork.outputLayer.perceptrons.map(x => x.output))
  }
}