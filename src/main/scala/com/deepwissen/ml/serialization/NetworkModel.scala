/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.serialization

import com.deepwissen.ml.algorithm.networks.{DeepNetwork, AutoencoderNetwork, Network, MarkovChain}
import com.deepwissen.ml.algorithm.HiddenLayer


/**
 * Created by hendri_k on 7/13/15.
 */
trait InferencesNetworkModel

/**
 * @author Eko Khannedy
 * @since 2/25/15
 */
case class NetworkModel(inputLayer: LayerModel,
                        hiddenLayers: List[LayerModel],
                        outputLayer: LayerModel,
                        synapsies: List[SynapsysModel]) extends InferencesNetworkModel

case class AutoencoderNetworkModel(inputLayer: LayerModel,
                                   hiddenLayer: LayerModel,
                                   outputLayer: LayerModel,
                                   synapsies: List[SynapsysModel]) extends InferencesNetworkModel


case class MarkovChainModel(inputLayer: LayerModel,
                        hiddenLayer: LayerModel,
                        synapsies: List[SynapsysModel]) extends InferencesNetworkModel

case class DeepNetworkModel(inputLayer: LayerModel,
                        hiddenLayers: List[LayerModel],
                        outputLayer: LayerModel,
                        synapsies: List[SynapsysModel]) extends InferencesNetworkModel


case class LayerModel(id: String,
                      perceptrons: List[PerceptronModel],
                      bias: Option[String],
                      biases : List[PerceptronModel] = null,
                      nextLayer: Option[String],
                      prevLayer: Option[String])

case class SynapsysModel(from: String,
                         to: String,
                         weight: Double,
                         deltaWeight: Double)

case class PerceptronModel(id: String,
                           index: Int,
                           output : Double = 0.0)

object NetworkModel {

  /**
   * Convert from Network to NetworkModel
   * @param network Network
   * @return NetworkModel
   */
  def apply(network: Network): NetworkModel = {

    val inputLayer = new LayerModel(
      id = network.inputLayer.id,
      perceptrons = network.inputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.inputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      nextLayer = network.inputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.inputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val hiddenLayers = network.hiddenLayers.map { layer =>
      LayerModel(
        id = layer.id,
        perceptrons = layer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
        bias = layer.bias.fold[Option[String]](None)(p => Some(p.id)),
        nextLayer = layer.next.fold[Option[String]](None)(l => Some(l.id)),
        prevLayer = layer.prev.fold[Option[String]](None)(l => Some(l.id))
      )
    }

    val outputLayer = new LayerModel(
      id = network.outputLayer.id,
      perceptrons = network.outputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.outputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      nextLayer = network.outputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.outputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val synapsies = network.synapsies.map { synapsys =>
      SynapsysModel(
        from = synapsys.from.id,
        to = synapsys.to.id,
        weight = synapsys.weight,
        deltaWeight = synapsys.deltaWeight
      )
    }

    NetworkModel(inputLayer, hiddenLayers, outputLayer, synapsies)
  }

}

/**
 * Created by hendri_k on 7/13/15.
 */
object AutoencoderNetworkModel {

  /**
   * Convert from Network to NetworkModel
   * @param network Network
   * @return NetworkModel
   */
  def apply(network: AutoencoderNetwork): AutoencoderNetworkModel = {

    val inputLayer = new LayerModel(
      id = network.inputLayer.id,
      perceptrons = network.inputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.inputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      nextLayer = network.inputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.inputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val hiddenLayers = new LayerModel(
        id = network.hiddenLayer.id,
        perceptrons = network.hiddenLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
        bias = network.hiddenLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
        nextLayer = network.hiddenLayer.next.fold[Option[String]](None)(l => Some(l.id)),
        prevLayer = network.hiddenLayer.prev.fold[Option[String]](None)(l => Some(l.id))
      )


    val outputLayer = new LayerModel(
      id = network.outputLayer.id,
      perceptrons = network.outputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.outputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      nextLayer = network.outputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.outputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val synapsies = network.synapsies.map { synapsys =>
      SynapsysModel(
        from = synapsys.from.id,
        to = synapsys.to.id,
        weight = synapsys.weight,
        deltaWeight = synapsys.deltaWeight
      )
    }

    AutoencoderNetworkModel(inputLayer, hiddenLayers, outputLayer, synapsies)
  }

}

/**
 * Created by hendri_k on 7/13/15.
 */
object MarkovChainModel {

  /**
   * Convert from Network to NetworkModel
   * @param network Network
   * @return NetworkModel
   */
  def apply(network: MarkovChain): MarkovChainModel = {

    val inputLayer = new LayerModel(
      id = network.inputLayer.id,
      perceptrons = network.inputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.inputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      biases = network.inputLayer.biases.map(p => PerceptronModel(p.id, p.index, p.output)),
      nextLayer = network.inputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.inputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val hiddenLayer = new LayerModel(
      id = network.hiddenLayer.id,
      perceptrons = network.hiddenLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.hiddenLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      biases = network.hiddenLayer.biases.map(p => PerceptronModel(p.id, p.index, p.output)),
      nextLayer = network.hiddenLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.hiddenLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val synapsies = network.synapsies.map { synapsys =>
      SynapsysModel(
        from = synapsys.from.id,
        to = synapsys.to.id,
        weight = synapsys.weight,
        deltaWeight = synapsys.deltaWeight
      )
    }

    MarkovChainModel(inputLayer, hiddenLayer, synapsies)
  }

}


object DeepNetworkModel {

  /**
   * Convert from Network to NetworkModel
   * @param network Network
   * @return NetworkModel
   */
  def apply(network: DeepNetwork): NetworkModel = {

    val inputLayer = new LayerModel(
      id = network.inputLayer.id,
      perceptrons = network.inputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.inputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      nextLayer = network.inputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.inputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val hiddenLayers = network.hiddenLayers.map { layer =>
      LayerModel(
        id = layer.id,
        perceptrons = layer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
        bias = layer.bias.fold[Option[String]](None)(p => Some(p.id)),
        nextLayer = layer.next.fold[Option[String]](None)(l => Some(l.id)),
        prevLayer = layer.prev.fold[Option[String]](None)(l => Some(l.id))
      )
    }

    val outputLayer = new LayerModel(
      id = network.outputLayer.id,
      perceptrons = network.outputLayer.perceptrons.map(p => PerceptronModel(p.id, p.index)),
      bias = network.outputLayer.bias.fold[Option[String]](None)(p => Some(p.id)),
      nextLayer = network.outputLayer.next.fold[Option[String]](None)(l => Some(l.id)),
      prevLayer = network.outputLayer.prev.fold[Option[String]](None)(l => Some(l.id))
    )

    val synapsies = network.synapsies.map { synapsys =>
      SynapsysModel(
        from = synapsys.from.id,
        to = synapsys.to.id,
        weight = synapsys.weight,
        deltaWeight = synapsys.deltaWeight
      )
    }

    NetworkModel(inputLayer, hiddenLayers, outputLayer, synapsies)
  }

}