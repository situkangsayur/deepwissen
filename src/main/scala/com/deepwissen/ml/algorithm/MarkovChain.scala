package com.deepwissen.ml.algorithm

import com.deepwissen.ml.serialization.{LayerModel, MarkovChainModel, NetworkModel, PerceptronModel}

import scala.annotation.tailrec

/**
 * Created by hendri_k on 7/8/15.
 */
class MarkovChain (var inputLayer: Layer,
                   var hiddenLayer: Layer,
                   var synapsies: List[Synapsys]) extends InferencesNetwork{

  @transient
  private val allPerceptrons: Map[String, Perceptron] =
    ((inputLayer.bias.get :: inputLayer.perceptrons)  :::
      (hiddenLayer.bias.get :: hiddenLayer.perceptrons)).map(p => (p.id, p)).toMap

  /**
   * Get perceptron by id
   * @param perceptronId perceptron id
   * @return Perceptron
   */
  def getPerceptron(perceptronId: String): Perceptron = allPerceptrons(perceptronId)

  @transient
  private val synapsiesLookupPair: Map[(String, String), Synapsys] = synapsies.map { synapsys =>
    ((synapsys.from.id, synapsys.to.id), synapsys)
  }.toMap

  /**
   * Get synapsys that go from source perceptron to target perceptron
   * @param fromPerceptronId source perceptron
   * @param toPerceptronId target perceptron
   * @return synapsys
   */
  def getSynapsys(fromPerceptronId: String, toPerceptronId: String) =
    synapsiesLookupPair(fromPerceptronId -> toPerceptronId)

  @transient
  private val synapsiesLookupFrom: Map[String, Seq[Synapsys]] = synapsies.groupBy(_.from.id)

  /**
   * Get all synapsies that go from source perceptron
   * @param perceptronId source perceptron
   * @return list of synapsies
   */
  def getSynapsiesFrom(perceptronId: String):Seq[Synapsys] = synapsiesLookupFrom(perceptronId)

  @transient
  private val synapsiesLookupTo: Map[String, Seq[Synapsys]] = synapsies.groupBy(_.to.id)

  /**
   * Get all synapsies that go to target perceptron
   * @param perceptronId target perceptron
   * @return list of synapsies
   */
  def getSynapsiesTo(perceptronId: String):Seq[Synapsys] = synapsiesLookupTo(perceptronId)

  /**
   * Get perceptron weight, calculate from all synapsies and perceptron source
   * @param perceptron perceptron
   * @return weight
   */
  def getPerceptronWeightTo(perceptron: Perceptron): Double =
    getSynapsiesTo(perceptron.id).foldLeft(0.0) { (value, synapsys) =>
      value + (synapsys.weight * synapsys.from.output)
    }

  def getPerceptronWeightFrom(perceptron: Perceptron): Double =
    getSynapsiesFrom(perceptron.id).foldLeft(0.0) { (value, synapsys) =>
      value + (synapsys.weight * synapsys.from.output)
    }
}


object MarkovChain{
  /**
   * Create new Perceptron from given Model
   * @param model Perceptron model
   * @return Perceptron
   */
  def newPerceptron(model: PerceptronModel) = Perceptron(model.id, model.index)

  /**
   * Create new Perceptron with unique id
   * @return Perceptron
   */
  def newPerceptron(index: Int) = Perceptron(newPerceptronId(), index)

  /**
   * Create new Bias with unique id
   * @return Perceptron of Bias
   */
  def newBias() = Perceptron(newBiasId(), -1)

  /**
   * Create new Bias with given id
   * @param id bias id
   * @return Perceptron of Bias
   */
  def newBias(id: String) = Perceptron(id, -1)

  /**
   * Create list of perceptron with given size
   * @param size size of perceptron
   * @return list
   */
  def newPerceptrons(size: Int) =
    (0 until size).map { i =>
      newPerceptron(i)
    }.toList

  /**
   * Create new Network with given perceptron input size and hidden layer size
   * @param inputPerceptronSize input perceptron size
   * @return
   */
  def apply(inputPerceptronSize: Int, outputPerceptronSize : Int,synapsysFactory: SynapsysFactory): MarkovChain = {
    val hiddenPerceptronSize = Math.round(inputPerceptronSize * 2 / 3.0).toInt

    // create input layer
    val inputLayer = InputLayer(
      id = newLayerId(),
      perceptrons = newPerceptrons(inputPerceptronSize),
      bias = Some(newBias())
    )

    var prevLayer: Layer = inputLayer

    // create output layer
    val hiddenLayer = new HiddenLayer(
      id = newLayerId(),
      perceptrons = newPerceptrons(outputPerceptronSize),
      bias = Some(newBias()),
      prev = Some(inputLayer)
    )
    prevLayer.next = Some(hiddenLayer)

    // create synapsies
    @tailrec
    def createSynapsies(prevLayer: Layer, synapsies: List[Synapsys]): List[Synapsys] =
      prevLayer.next match {
        case None => synapsies // no next layer
        case Some(nextLayer) =>
          //  prev perceptron + bias
          val prevPerceptrons = prevLayer.bias.get :: prevLayer.perceptrons
          // next perceptron - bias
          val nextPerceptrons = nextLayer.perceptrons
          // create perceptrom from prev layer to next layer
          val currentSynapsies = prevPerceptrons.flatMap { prevPerceptron =>
            nextPerceptrons.map { nextPerceptron =>
              synapsysFactory(prevPerceptron, nextPerceptron)
            }
          }
          // go to next layer
          createSynapsies(nextLayer, synapsies ::: currentSynapsies)
      }

    val synapsies = createSynapsies(inputLayer, List())

    // create network
    new MarkovChain(inputLayer, hiddenLayer, synapsies)
  }

  /**
   * Convert from NetworkModel to Network
   * @param model NetworkModel
   * @return Network
   */
  def apply(model: MarkovChainModel): MarkovChain = {

    // create input layer
    val inputLayer = InputLayer(
      id = model.inputLayer.id,
      perceptrons = model.inputLayer.perceptrons.map(newPerceptron).sortBy(_.index),
      bias = model.inputLayer.bias.fold[Option[Perceptron]](None)(id => Some(newBias(id)))
    )

    // create hidden layer
//    val hiddenLayers = model.hiddenLayers.map { layer =>
//      HiddenLayer(
//        id = layer.id,
//        perceptrons = layer.perceptrons.map(newPerceptron).sortBy(_.index),
//        bias = layer.bias.fold[Option[Perceptron]](None)(id => Some(newBias(id)))
//      )
//    }

    // create output layer
    val hiddenLayer = new HiddenLayer(
      id = model.hiddenLayer.id,
      perceptrons = model.hiddenLayer.perceptrons.map(newPerceptron).sortBy(_.index),
      bias = model.hiddenLayer.bias.fold[Option[Perceptron]](None)(id => Some(newBias(id)))
    )

    // crate layer relation
    val allLayers = List(inputLayer, hiddenLayer)
    val allModelLayers = List(model.inputLayer , model.hiddenLayer)

    def findLayerModel(id: String) = allModelLayers.find(_.id == id)
    def findLayer(id: String) = allLayers.find(_.id == id)

    allLayers.foreach { currentLayer =>
      currentLayer.next = findLayerModel(currentLayer.id).fold[Option[Layer]](None)({ layerModel =>
        layerModel.nextLayer.fold[Option[Layer]](None)(nextLayerId => {
          findLayer(nextLayerId)
        })
      })

      currentLayer.prev = findLayerModel(currentLayer.id).fold[Option[Layer]](None)({ layerModel =>
        layerModel.prevLayer.fold[Option[Layer]](None)(prevLayerId => {
          findLayer(prevLayerId)
        })
      })
    }

    // all perceptron in network
    val allPerceptrons = ((inputLayer.bias.get :: inputLayer.perceptrons) :::
      (hiddenLayer.bias.get :: hiddenLayer.perceptrons)).map(p => (p.id, p)).toMap

    // create synapsies
    val synapsies = model.synapsies.map { synapsys =>
      Synapsys(
        from = allPerceptrons(synapsys.from),
        to = allPerceptrons(synapsys.to),
        weight = synapsys.weight,
        deltaWeight = synapsys.deltaWeight
      )
    }

    new MarkovChain(inputLayer, hiddenLayer, synapsies)
  }
}