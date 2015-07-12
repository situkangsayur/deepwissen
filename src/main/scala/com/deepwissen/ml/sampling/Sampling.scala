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
//    val network = newNetwork(dataset, parameter)
//    doTrain(network, dataset, parameter)
//    network
    null
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
  def newNetwork(dataset: DATASET, parameter: GibbsParameter): MarkovChain= null
//    MarkovChain(
//      inputPerceptronSize = parameter.inputPerceptronSize,
//      outputPerceptronSize = parameter.inputPerceptronSize,
//      synapsysFactory = parameter.synapsysFactory
//    )

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

//      case Some(nextLayer) =>
//        // hidden or input layer
//        val sigmaError = nextLayer.perceptrons.foldLeft(0.0) { (value, toPerceptron) =>
//          value + toPerceptron.error * network.getSynapsys(fromPerceptron.id, toPerceptron.id).weight
//        }
//        fromPerceptron.output * (1 - fromPerceptron.output) * sigmaError
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
//    if (synapsys.isFromBias)
//      (parameter.learningRate * perceptron.error) + (parameter.momentum * synapsys.deltaWeight)
//    else
//      (parameter.learningRate * perceptron.error * synapsys.from.output) + (parameter.momentum * synapsys.deltaWeight)
    0.0D
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

    /**
     * Update weight and output for all hidden layers
     */
//    network.hiddenLayers.foreach { layer =>
//      layer.perceptrons.foreach { perceptron =>
//        perceptron.weight = network.getPerceptronWeight(perceptron)
//        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
//      }
//    }

    /**
     * Update weight and output for output layer
     */
//    network.outputLayer.perceptrons.foreach { perceptron =>
//      perceptron.weight = network.getPerceptronWeight(perceptron)
//      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
//    }

    /**
     * Update error for output layer
     */
    network.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.error = getPerceptronError(network, network.outputLayer, perceptron, data, parameter)
    }

    /**
     * Update errors for all hidden layers
     */
//    network.hiddenLayers.reverse.foreach { layer =>
//      layer.perceptrons.foreach { perceptron =>
//        perceptron.error = getPerceptronError(network, layer, perceptron, data, parameter)
//      }
//    }

    /**
     * Update weight all synapsies to hidden layers
     */
//    network.hiddenLayers.foreach { layer =>
//      layer.perceptrons.foreach { perceptron =>
//        network.getSynapsiesTo(perceptron.id).foreach { synapsys =>
//          synapsys.deltaWeight = getSynapsysDeltaWeight(network, perceptron, synapsys, parameter)
//          synapsys.weight = synapsys.weight + synapsys.deltaWeight
//        }
//      }
//    }

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
      value + Math.pow(getTargetClass(data, perceptron.index).get - perceptron.output, 2)
    }
    sumError / network.outputLayer.perceptrons.length
  }
}