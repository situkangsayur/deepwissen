package com.deepwissen.ml.sampling

import com.deepwissen.ml.algorithm._
import com.deepwissen.ml.utils.{ContValue, Denomination}

/**
 * Created by hendri_k on 7/7/15.
 */
trait Sampling[T] {
  def doSampling : T
}


abstract class GibbSampling[DATASET] extends Sampling[Layer] with Algorithm[DATASET, Array[Double], GibbsParameter, MarkovChain]{

  /**
   * Run training with given dataset
   * @param dataset dataset
   * @param parameter parameter
   * @return model
   */
  override def train(dataset: DATASET, parameter: GibbsParameter): MarkovChain = {
    val network = newNetwork(dataset, parameter)
    doTrain(network, dataset, parameter)
    network
  }

  override def doSampling : Layer ={
    null
  }



//  ==========================

  /**n
    * Create new network from dataset and training parameter
    * @param dataset dataset
    * @param parameter parameter
    * @return network
    */
  def newNetwork(dataset: DATASET, parameter: GibbsParameter): MarkovChain =
    MarkovChain(
      inputPerceptronSize = parameter.inputPerceptronSize,
      outputPerceptronSize = parameter.hiddenPerceptronSize,
      synapsysFactory = parameter.synapsysFactory
    )

  /**
   * Train implementation
   * @param network network
   * @param dataset dataset
   * @param parameter training parameter
   */
  def doTrain(network: MarkovChain, dataset: DATASET, parameter: GibbsParameter): Unit

  /**
   * Get target class
   * @param data data
   * @return
   */
  def getTargetClass(data: Array[Denomination[_]], index: Int) = data(index).asInstanceOf[ContValue]

  /**
   * Get perceptron error calculation
   * @param network network
   * @param layer layer
   * @param fromPerceptron perceptron
   * @param data dataset
   * @param parameter train parameter
   * @return error
   */
  def getPerceptronError(network: MarkovChain, layer: Layer, fromPerceptron: Perceptron, data: Array[Denomination[_]], parameter: TrainingParameter): Double = {
    layer.next match {
      case None =>
        // output layer
        fromPerceptron.output * (1 - fromPerceptron.output) * (getTargetClass(data,fromPerceptron.index).get - fromPerceptron.output)

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
  def getSynapsysDeltaWeight(network: MarkovChain, perceptron: Perceptron, synapsys: Synapsys, parameter: GibbsParameter): Double = {
    if (synapsys.isFromBias)
      (parameter.learningRate * perceptron.error) + (parameter.momentum * synapsys.deltaWeight)
    else
      (parameter.learningRate * perceptron.error * synapsys.from.output) + (parameter.momentum * synapsys.deltaWeight)
//    0.0D
  }

  /**
   * Train network model with single data from dataset
   * @param data data
   * @param network network model
   * @param parameter train parameter
   * @return sum error
   */
  def doTrainData(data: Array[Denomination[_]], network: MarkovChain, parameter: GibbsParameter): Double = {

    /**
     * Update output for all input layer
     */
    network.inputLayer.fillOutput(data)



    var originalData = InputLayer(
      id = network.inputLayer.id,
      perceptrons = network.inputLayer.perceptrons.map( x => Perceptron(x.id,x.index,x.output, x.weight, x.error)),
      bias = Some(MarkovChain.newBias(network.inputLayer.bias.get.id))
    )
    originalData.bias.get.output = network.inputLayer.bias.get.output
    originalData.bias.get.weight = network.inputLayer.bias.get.weight

//    originalData.fillOutput(data)

    /**
     * Update weight and output for all hidden layers
     */
//    network.hiddenLayers.foreach { layer =>
//      layer.perceptrons.foreach { perceptron =>
//        perceptron.weight = network.getPerceptronWeight(perceptron)
//        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
//      }
//    }

    for(i <- 1 to parameter.k){
      /**
       * Update weight and output for output layer
       */
      network.hiddenLayer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightTo(perceptron)
        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
      }

      /**
       * update input layer
       */
      network.inputLayer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightFrom(perceptron)
        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
      }
    }

    /**
     * Update weight and output for output layer
     */
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }


    var tempXTilt = InputLayer(
      id = network.inputLayer.id,
      perceptrons = network.inputLayer.perceptrons.map( x => Perceptron(x.id,x.index,x.output, x.weight, x.error)),
      bias = Some(MarkovChain.newBias(network.inputLayer.bias.get.id))
    )

    tempXTilt.bias.get.output = network.inputLayer.bias.get.output
    tempXTilt.bias.get.weight = network.inputLayer.bias.get.weight

    var hXTotal = 0.0D;
    var hxTiltTotal = 0.0D;

    network.hiddenLayer.perceptrons.foreach { perceptron =>

      val hXTilt = perceptron.output
      hxTiltTotal += hXTilt

      network.inputLayer = originalData
      val hX = parameter.activationFunction.activation(network.getPerceptronWeightTo(perceptron))
      hXTotal += hX

      network.getSynapsiesTo(perceptron.id).foreach { synapsys =>

        synapsys.deltaWeight = parameter.learningRate * ((hX * originalData.perceptrons.find( p => p.id == synapsys.from.id).get.output)
          - (hXTilt * tempXTilt.perceptrons.find( p => p.id == synapsys.from.id).get.output))
        synapsys.weight = synapsys.weight + synapsys.deltaWeight
      }
    }

    network.hiddenLayer.bias.get.output = network.hiddenLayer.bias.get.output + (parameter.learningRate *
      (hXTotal/network.hiddenLayer.perceptrons.size) - (hxTiltTotal/network.hiddenLayer.perceptrons.size))

    network.inputLayer.bias.get.output = network.hiddenLayer.bias.get.output + (parameter.learningRate *
      (originalData.perceptrons.zip(tempXTilt.perceptrons).map(p => (p._1.output - p._2.output)).foldLeft(0.0D)((temp, x) => temp + x))/originalData.perceptrons.size)

    network.inputLayer = tempXTilt

    /**
     * Update weight and output for output layer
     */
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }

    /**
     * Lost function before partial funtion adding
     */
    val freeEnergy = Math.exp(network.inputLayer.perceptrons.map( p => p.output * network.hiddenLayer.bias.get.output).foldLeft(0.0D)((temp, value) => temp + value) +
    network.hiddenLayer.perceptrons.map(p => Math.log(1 + p.output)).foldLeft(0.0D)((temp, value) => temp + value))

    freeEnergy
  }
}