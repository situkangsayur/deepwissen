package com.deepwissen.ml.algorithm

import com.deepwissen.ml.utils.{ContValue, Denomination}

/**
 * Created by hendri_k on 7/7/15.
 */
abstract class AbstractRestrictedBoltzmannMachine[DATASET] extends Algorithm[DATASET, Array[Double], GibbsParameter, MarkovChain]{

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
    val oldNetwork = MarkovChain(network)

    oldNetwork.inputLayer.perceptrons.foreach( p => print(p.id+"-"+p.output+"; "))
    println()
    println()
    oldNetwork.hiddenLayer.perceptrons.foreach( p => print(p.id+"-"+p.output+"; "))

    println()
    println()

    oldNetwork.synapsies.foreach(p => print(p.from.id + " - "+p.to.id + " - " + p.weight+"; \n"))
    println()
    println()

    for(i <- 1 to parameter.k){
      println("===========================================================================================>>" + i)
      println("output hidden layer ")
      /**
       * Update weight and output for output layer
       */
      network.hiddenLayer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightTo(perceptron)
        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
        print(perceptron.output + ",")
      }
      println()

      println("input balik : ")

      /**
       * update input layer
       */
      network.inputLayer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightFrom(perceptron)
        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
        print(perceptron.output + ",")
      }
      println("next iteration ------->>")
    }

    oldNetwork.hiddenLayer.perceptrons.foreach { perceptron =>
      println(perceptron)
      perceptron.weight = oldNetwork.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
      print("original data : "+perceptron.output + ",")
    }

    println()

    /**
     * Update weight and output for output layer
     */
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
    }


    println("k : " + parameter.k)
    println("original ::")
    oldNetwork.inputLayer.perceptrons.foreach( p => print(p.output + "," + p.id + "; "))
    println()
    println()

    println("x tilt  ::")
    network.inputLayer.perceptrons.foreach( p => print(p.output + "," + p.id + "; "))
    println()
    println()

    println("original ::")
    network.synapsies.foreach( p => print(p.from.output +"-"+p.from.id+";" + p.to.output+"-"+p.to.id+"; ---- \n"))
    println()
    println()


    println("original ::")
    oldNetwork.inputLayer.perceptrons.foreach( p => print(p.output + ","))
    println()

    var hXTotal = 0.0D
    var hxTiltTotal = 0.0D

    network.hiddenLayer.perceptrons.foreach { perceptron =>

      print("X-tilt :: ")
      print(perceptron.output)
      println()
      val hXTilt = perceptron.output
      hxTiltTotal += hXTilt

      println("get act function hX")
      val hX = oldNetwork.hiddenLayer.perceptrons.find(p => p.id.equals(perceptron.id)).get.output
      hXTotal += hX
      print("hX :: ")
      print(hX)
      println()

      println("original ::")
      oldNetwork.inputLayer.perceptrons.foreach( p => print(p + ","))
      println()

      println("====="+network.getSynapsiesTo(perceptron.id)+"-----")

      network.getSynapsiesTo(perceptron.id).foreach { synapsys =>
//        println(":>>>"+synapsys.from)
//        println(":>> "+(oldNetwork.inputLayer.bias.get::oldNetwork.inputLayer.perceptrons).find( p => p.id.equals(synapsys.from.id)).get)
        synapsys.deltaWeight = parameter.learningRate * ((hX *
          (oldNetwork.inputLayer.bias.get::oldNetwork.inputLayer.perceptrons).find( p => p.id.equals(synapsys.from.id)).get.output)
          - (hXTilt * (network.inputLayer.bias.get::network.inputLayer.perceptrons).find( p => p.id.equals(synapsys.from.id)).get.output))
        synapsys.weight = synapsys.weight + synapsys.deltaWeight
      }
    }

    network.hiddenLayer.bias.get.output = network.hiddenLayer.bias.get.output + (parameter.learningRate *
      (hXTotal/network.hiddenLayer.perceptrons.size) - (hxTiltTotal/network.hiddenLayer.perceptrons.size))

    network.inputLayer.bias.get.output = network.hiddenLayer.bias.get.output + (parameter.learningRate *
      (oldNetwork.inputLayer.perceptrons.zip(network.inputLayer.perceptrons).map(p => (p._1.output - p._2.output)).foldLeft(0.0D)((temp, x) => temp + x))/
      oldNetwork.inputLayer.perceptrons.size)



    /**
     * Lost function before partial funtion adding
     */
    val freeEnergy = Math.exp(network.inputLayer.perceptrons.map( p => p.output * network.hiddenLayer.bias.get.output).foldLeft(0.0D)((temp, value) => temp + value) +
    network.hiddenLayer.perceptrons.map(p => Math.log(1 + p.output)).foldLeft(0.0D)((temp, value) => temp + value))

    println("free energy : " + freeEnergy)
    freeEnergy
  }
}