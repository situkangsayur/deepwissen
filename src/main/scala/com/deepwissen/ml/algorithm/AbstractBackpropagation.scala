/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * Abstract implementation of Neural Network Backpropagation
 * @author Eko Khannedy
 * @since 3/14/15
 */
abstract class AbstractBackpropagation[DATASET] extends Algorithm[DATASET, Array[Double], BackpropragationParameter, Network] {

  /**
   * Get perceptron weight, calculate from all synapsies and perceptron source
   * @param network network
   * @param perceptron perceptron
   * @return weight
   */
  def getPerceptronWeight(network: Network, perceptron: Perceptron): Double = {
    val synapsies = network.getSynapsiesTo(perceptron.id)
    val weight = synapsies.foldLeft(0.0) { (value, synapsys) =>
      value + (synapsys.weight * network.getPerceptron(synapsys.from).output)
    }
    weight
  }

  /**
   * Get target class
   * @param data data
   * @return
   */
  def getTargetClass(data: Array[Double]) = data(data.length - 1)

  /**
   * Get perceptron error calculation
   * @param network network
   * @param layer layer
   * @param fromPerceptron perceptron
   * @param data dataset
   * @param parameter train parameter
   * @return error
   */
  def getPerceptronError(network: Network, layer: Layer, fromPerceptron: Perceptron, data: Array[Double], parameter: TrainingParameter): Double = {
    layer.next match {
      case None =>
        // output layer
        fromPerceptron.output * (1 - fromPerceptron.output) * (getTargetClass(data) - fromPerceptron.output)

      case Some(nextLayer) =>
        // hidden or input layer
        val sigmaError = nextLayer.perceptrons.foldLeft(0.0) { (value, toPerceptron) =>
          value + toPerceptron.error * network.getSynapsys(fromPerceptron.id, toPerceptron.id).weight
        }
        fromPerceptron.output * (1 - fromPerceptron.output) * sigmaError
    }
  }

  /**
   * Get synapsys delta weight calculation
   * @param network network
   * @param perceptron perceptron
   * @param synapsys synapsys
   * @param parameter parameter
   * @return delta weight
   */
  def getSynapsysDeltaWeight(network: Network, perceptron: Perceptron, synapsys: Synapsys, parameter: BackpropragationParameter): Double = {
    if (synapsys.isFromBias)
      (parameter.learningRate * perceptron.error) + (parameter.momentum * synapsys.deltaWeight)
    else
      (parameter.learningRate * perceptron.error * network.getPerceptron(synapsys.from).output) + (parameter.momentum * synapsys.deltaWeight)
  }

  /**
   * Train network model with single data from dataset
   * @param data data
   * @param network network model
   * @param parameter train parameter
   * @return sum error
   */
  def doTrainData(data: Array[Double], network: Network, parameter: BackpropragationParameter): Double = {

    /**
     * Update output for all input layer
     */
    network.inputLayer.fillOutput(data)

    /**
     * Update weight and output for all hidden layers
     */
    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        perceptron.weight = getPerceptronWeight(network, perceptron)
        perceptron.output = perceptron.activationFunction.activation(perceptron.weight)
      }
    }

    /**
     * Update weight and output for output layer
     */
    network.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = getPerceptronWeight(network, perceptron)
      perceptron.output = perceptron.activationFunction.activation(perceptron.weight)
    }

    /**
     * Update error for output layer
     */
    network.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.error = getPerceptronError(network, network.outputLayer, perceptron, data, parameter)
    }

    /**
     * Update errors for all hidden layers
     */
    network.hiddenLayers.reverse.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        perceptron.error = getPerceptronError(network, layer, perceptron, data, parameter)
      }
    }

    /**
     * Update weight all synapsies to hidden layers
     */
    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        network.getSynapsiesTo(perceptron.id).foreach { synapsys =>
          synapsys.deltaWeight = getSynapsysDeltaWeight(network, perceptron, synapsys, parameter)
          synapsys.weight = synapsys.weight + synapsys.deltaWeight
        }
      }
    }

    /**
     * Update weight all synapsies to output layer
     */
    network.outputLayer.perceptrons.foreach { perceptron =>
      network.getSynapsiesTo(perceptron.id).foreach { synapsys =>
        synapsys.deltaWeight = getSynapsysDeltaWeight(network, perceptron, synapsys, parameter)
        synapsys.weight = synapsys.weight + synapsys.deltaWeight
      }
    }

    /**
     * Sum squared error
     */
    val sumError = network.outputLayer.perceptrons.foldLeft(0.0) { (value, perceptron) =>
      value + Math.pow(getTargetClass(data) - perceptron.output, 2)
    }
    sumError / network.outputLayer.perceptrons.length
  }

  /**
   * Classification for given data with given network model
   * @param data data
   * @param network network model
   * @return double
   */
  override def classification(data: Array[Double], network: Network): Double = {
    // fill input layer
    network.inputLayer.fillOutput(data)

    // fill hidden layer
    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        perceptron.weight = getPerceptronWeight(network, perceptron)
        perceptron.output = perceptron.activationFunction.activation(perceptron.weight)
      }
    }

    // fill output layer
    network.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = getPerceptronWeight(network, perceptron)
      perceptron.output = perceptron.activationFunction.activation(perceptron.weight)
    }

    // calculate result
    network.outputLayer.perceptrons.foldLeft(0.0) { (value, perceptron) =>
      value + perceptron.output
    }
  }

}