/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.algorithm.networks.Network
import com.deepwissen.ml.utils.{LogPrint, BinaryValue, Denomination}

/**
 * Abstract implementation of Neural Network Backpropagation
 * @author Eko Khannedy
 * @since 3/14/15
 */
abstract class AbstractBackpropagation[DATASET] extends Algorithm[DATASET, Array[Double], BackpropragationParameter, Network] {

  /**
   * Run training with given dataset
   * @param dataset dataset
   * @param parameter parameter
   * @return model
   */
  override def train(dataset: DATASET, parameter: BackpropragationParameter): Network = {
    val network = newNetwork(dataset, parameter)
    doTrain(network, dataset, parameter)
    network
  }

  /**
   * Create new network from dataset and training parameter
   * @param dataset dataset
   * @param parameter parameter
   * @return network
   */
  def newNetwork(dataset: DATASET, parameter: BackpropragationParameter): Network =
    Network(
      inputPerceptronSize = parameter.inputPerceptronSize,
      hiddenSize = parameter.hiddenLayerSize,
      hiddenNodeSize = parameter.hiddenNodeSize,
      outputPerceptronSize = parameter.outputPerceptronSize,
      synapsysFactory = parameter.synapsysFactory
    )

  /**
   * Train implementation
   * @param network network
   * @param dataset dataset
   * @param parameter training parameter
   */
  def doTrain(network: Network, dataset: DATASET, parameter: BackpropragationParameter): Unit

  /**
   * Get target class
   * @param data data
   * @return
   */
  def getTargetClass(data: Array[Denomination[_]], targetClass: Int):BinaryValue =
    if(targetClass == -1) data(data.length - 1).asInstanceOf[BinaryValue] else data(targetClass).asInstanceOf[BinaryValue]

  /**
   * Get perceptron error calculation
   * @param network network
   * @param layer layer
   * @param fromPerceptron perceptron
   * @param data dataset
   * @param parameter train parameter
   * @return error
   */
  def getPerceptronError(network: Network, layer: Layer, fromPerceptron: Perceptron, data: Array[Denomination[_]], parameter: BackpropragationParameter): Double = {

    layer.next match {
      case None =>
        val tempError = fromPerceptron.output * (1 - fromPerceptron.output) * (getTargetClass(data, parameter.targetClassPosition).get(fromPerceptron.index) - fromPerceptron.output)
        LogPrint.printLogDebug("Output layeroutput --> output " + fromPerceptron.index + " :> "+ fromPerceptron.output + " - " + getTargetClass(data, parameter.targetClassPosition).get(fromPerceptron.index) + " = "+ tempError)

        // output layer
        tempError

      case Some(nextLayer) =>
        // hidden or input layer
        val sigmaError = nextLayer.perceptrons.foldLeft(0.0) { (value, toPerceptron) =>
          value + toPerceptron.error * network.getSynapsys(fromPerceptron.id, toPerceptron.id).weight
        }
        val tempErrorHidden = fromPerceptron.output * (1 - fromPerceptron.output) * sigmaError
        LogPrint.printLogDebug("output layer ---> output " + fromPerceptron.index + " :> " + tempErrorHidden)
        tempErrorHidden
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
      (parameter.learningRate * perceptron.error * synapsys.from.output) + (parameter.momentum * synapsys.deltaWeight)
  }

  /**
   * Train network model with single data from dataset
   * @param data data
   * @param network network model
   * @param parameter train parameter
   * @return sum error
   */
  def doTrainData(data: Array[Denomination[_]], network: Network, parameter: BackpropragationParameter): Double = {

    /**
     * Update output for all input layer
     */
    network.inputLayer.fillOutput(data)


    /**
     * Update weight and output for all hidden layers
     */
    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        val tempWeight = network.getPerceptronWeightTo(perceptron)
//        parameter.activationFunction.activation(perceptron.weight)
        val tempOutput = parameter.activationFunction.activation(tempWeight)
        LogPrint.printLogDebug("hidden layer :> "+ layer.id + " --> "+ perceptron.index + " :> " + tempWeight + " & " + tempOutput)
        perceptron.weight = tempWeight
        perceptron.output = tempOutput
      }
    }

    /**
     * Update weight and output for output layer
     */
    network.outputLayer.perceptrons.foreach { perceptron =>
      val tempWeight = network.getPerceptronWeightTo(perceptron)
//      parameter.activationFunction.activation(perceptron.weight)
      val tempOutput = parameter.activationFunction.activation(tempWeight)
      LogPrint.printLogDebug("Output layer :> " + " --> "+ perceptron.index + " :> " + tempWeight + " & " + tempOutput)
      perceptron.weight = tempWeight
      perceptron.output = tempOutput
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
      value + Math.pow(getTargetClass(data, parameter.targetClassPosition).get(perceptron.index) - perceptron.output, 2)
    }
//    val tempSumOfError = sumError / network.outputLayer.perceptrons.length
    val tempSumOfError = sumError
    LogPrint.printLogDebug("<--------------------------->"+tempSumOfError+"<--------------------------->")
    tempSumOfError
  }

}
