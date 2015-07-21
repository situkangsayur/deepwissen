package com.deepwissen.ml.algorithm

import com.deepwissen.ml.utils.{ContValue, Denomination}

/**
 * Created by hendri_k on 7/7/15.
 */
abstract class AbstractRestrictedBoltzmannMachine[DATASET] extends Algorithm[DATASET, Array[Double], GibbsParameter, MarkovChain]{

  var xTiltBefore: List[Perceptron] = null
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

  /**
    * Create new network from dataset and training parameter
    * @param dataset dataset
    * @param parameter parameter
    * @return network
    */
  def newNetwork(dataset: DATASET, parameter: GibbsParameter): MarkovChain =
    MarkovChain(
      inputPerceptronSize = parameter.inputPerceptronSize,
      hiddenPerceptronSize = parameter.hiddenPerceptronSize,
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

//  /**
//   * Get synapsys delta weight calculation
//   * @param network network
//   * @param perceptron perceptron
//   * @param synapsys synapsys
//   * @param parameter parameter
//   * @return delta weight
//   */
//  def getSynapsysDeltaWeight(network: MarkovChain, perceptron: Perceptron, synapsys: Synapsys, parameter: GibbsParameter): Double = {
//    if (synapsys.isFromBias)
//      (parameter.learningRate * perceptron.error) + (parameter.momentum * synapsys.deltaWeight)
//    else
//      (parameter.learningRate * perceptron.error * synapsys.from.output) + (parameter.momentum * synapsys.deltaWeight)
////    0.0D
//  }

  /**
   * Train network model with single data from dataset
   * @param data data
   * @param network network model
   * @param parameter train parameter
   * @return sum error
   */
  def doTrainData(data: Array[Denomination[_]], xTiltParam: List[Perceptron] , network: MarkovChain, parameter: GibbsParameter): (Double, List[Perceptron]) = {

    /**
     * Update output for all input layer
     */

    if(xTiltParam == null)
      network.inputLayer.fillOutput(data)
    else
      network.inputLayer.fillOutput(xTiltParam)

    var oldNetwork = MarkovChain(network, synapsysFactory = SetSynapsysFactory())

    oldNetwork.inputLayer.fillOutput(data)


    oldNetwork.hiddenLayer.biases.foreach { perceptron =>
      perceptron.output = network.getBias(perceptron.id).output
      print(perceptron.output + ",")
    }
    println(";")

    oldNetwork.inputLayer.biases.foreach { perceptron =>
      perceptron.output = network.getBias(perceptron.id).output
      print(perceptron.output + ",")
    }
    println(";")

    oldNetwork.synapsies.foreach { s =>
      s.weight = network.getSynapsys(s.from.id, s.to.id).weight
    }


    /**
     * Gibbs sampling for k-step
     */
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

    oldNetwork.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = oldNetwork.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }

    println()

    /**
     * Update weight and output for output layer
     */
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }

    println()
    print("input: ")
    oldNetwork.inputLayer.perceptrons.foreach( p => print(p.output + "; "))
    println()
    print("xtilt: ")
    network.inputLayer.perceptrons.foreach( p => print(p.output + "; "))
    println()
    println()

    println("get update\n")
    network.hiddenLayer.perceptrons.foreach { perceptron =>

      val hXTilt = perceptron.output


      val hX = oldNetwork.hiddenLayer.perceptrons.find(p => p.id.equals(perceptron.id)).get.output

      println("X-tilt: " + hXTilt + " --> hX: " + hX+" | next | ")

      network.getSynapsiesTo(perceptron.id).foreach { synapsys =>
        synapsys.deltaWeight = parameter.learningRate * ((hX *
          oldNetwork.inputLayer.perceptrons.find( p => p.id.equals(synapsys.from.id)).get.output)
          - (hXTilt * network.inputLayer.perceptrons.find( p => p.id.equals(synapsys.from.id)).get.output))
        synapsys.weight = synapsys.weight + synapsys.deltaWeight
      }
    }

    println()

    network.hiddenLayer.biases.foreach { p =>
      p.output = p.output + (parameter.learningRate *
        oldNetwork.getPerceptron(p.id.replace("bias","perceptron")).output -
        network.getPerceptron(p.id.replace("bias","perceptron")).output)
    }
//    network.hiddenLayer.bias.get.output = network.hiddenLayer.biases +

    network.inputLayer.biases.foreach { p =>
      p.output = p.output + (oldNetwork.getPerceptron(p.id.replace("bias","perceptron")).output -
        network.getPerceptron(p.id.replace("bias","perceptron")).output)
    }

//    network.inputLayer.bias.get.output = network.hiddenLayer.bias.get.output + (parameter.learningRate *
//      (oldNetwork.inputLayer.perceptrons.zip(network.inputLayer.perceptrons).map(p => (p._1.output - p._2.output)).foldLeft(0.0D)((temp, x) => temp + x))/
//      oldNetwork.inputLayer.perceptrons.size)


    /**
     * update all real biases values
     */
    oldNetwork.hiddenLayer.biases.foreach { perceptron =>
      perceptron.output = network.getBias(perceptron.id).output
      print(perceptron.output + ",")
    }
    println(";")

    oldNetwork.inputLayer.biases.foreach { perceptron =>
      perceptron.output = network.getBias(perceptron.id).output
      print(perceptron.output + ",")
    }
    println(";\n")

    oldNetwork.synapsies.foreach { s =>
      s.weight = network.getSynapsys(s.from.id, s.to.id).weight
    }

    /**
     * Update weight and output for output layer
     */
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }

    oldNetwork.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }

//    println("After ==>> ")
//    oldNetwork.synapsies.foreach {s =>
//      println("synapsis: "+s.from + " -> "+ s.to +" = " + s.weight)
//    }

    /**
     * Lost function before partial funtion adding
     */
    val freeEnergy = Math.exp(oldNetwork.inputLayer.perceptrons.map( p => p.output *
      oldNetwork.getBias(p.id.replace("perceptron", "bias")).output).foldLeft(0.0D)((temp, value) => temp + value) +
      oldNetwork.hiddenLayer.perceptrons.map(p => Math.log(1 + p.output)).foldLeft(0.0D)((temp, value) => temp + value))

    println("free energy : " + freeEnergy)
    println("----------------------------------------------")
    (freeEnergy, network.inputLayer.perceptrons.map( p => Perceptron(p.id, p.index, p.output, p.weight, p.error)))
  }
}