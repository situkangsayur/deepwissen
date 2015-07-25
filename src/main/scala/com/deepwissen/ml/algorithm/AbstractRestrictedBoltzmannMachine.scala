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
   * Train network model with single data from dataset
   * @param data data
   * @param network network model
   * @param parameter train parameter
   * @return sum error
   */
  def doTrainData(data: Array[Denomination[_]], xTiltParam: List[Perceptron] , network: MarkovChain, parameter: GibbsParameter): (Double, List[Perceptron]) = {


    network.inputLayer.fillOutput(data)
    val dataX = network.inputLayer.perceptrons.map(p => Perceptron(p.id, p.index, p.output, p.weight, p.error))

    /**
     * print synapsis
     */
//    network.synapsies.foreach( s => println(s.from.id+"("+s.from.output+") - "+ s.to.id+"("+s.to.output+") = " + s.weight))
//    println("----------------------------------------------------------------------------------------------------------------")
    /**
     * Update output for all input layer
     */
    if(xTiltParam != null){
      network.inputLayer.fillOutput(xTiltParam)
//      network.updateInputLayerValues(xTiltParam)
    }

    /**
     * print synapsies after adding xTilt(t-1)
     */
//    network.synapsies.foreach( s => println(s.from.id+"("+s.from.output+") - "+ s.to.id+"("+s.to.output+") = " + s.weight))
//    println("================================================================================================================")

    /**
     * Gibbs sampling for k-step
     */
    for(i <- 1 to parameter.k){
      /**
       * Update weight and output for output layer
       */
      network.hiddenLayer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightTo(perceptron,false)
        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
        perceptron.sample = parameter.sampling.getValue(parameter.activationFunction.activation(perceptron.weight))
      }
      /**
       * update input layer
       */
      network.inputLayer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeightFrom(perceptron,false)
        perceptron.output = parameter.activationFunction.activation(perceptron.weight)
        perceptron.sample = parameter.sampling.getValue(parameter.activationFunction.activation(perceptron.weight))
      }
    }

    /**
     * Update output hidden Layer Xtilt
     */
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron, false)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
      perceptron.sample = parameter.sampling.getValue(parameter.activationFunction.activation(perceptron.weight))
    }

    val dataXTilt = network.inputLayer.perceptrons.map(p => Perceptron(p.id, p.index, p.output, p.weight, p.error, p.sample))

    val listOfOutputXTilt = network.hiddenLayer.perceptrons.map(p => Perceptron(p.id, p.index, p.output, p.weight, p.error, p.sample))

    /**
     * get Output for data X (real data)
     */
    network.updateInputLayerValues(dataX)
    network.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeightTo(perceptron,false)
      perceptron.output = parameter.activationFunction.activation(perceptron.weight)
      perceptron.sample = parameter.sampling.getValue(parameter.activationFunction.activation(perceptron.weight))
    }

    val listOfOutputX = network.hiddenLayer.perceptrons.map(p => Perceptron(p.id, p.index, p.output, p.weight, p.error, p.sample))

    /**
     * show input layer values
     */
//    println("X : ")
//    network.inputLayer.perceptrons.foreach( p => println(p.id+" : "+ p.output +" ; bias : " + network.getBias(p.id.replace("perceptron", "bias")).output))
//    println()
//    println("xtilt : ")
//    dataXTilt.foreach( p => println(p.id+" : "+ p.output + " ; bias : " + network.getBias(p.id.replace("perceptron", "bias")).output))
//    println()
//    println()


    /**
     * show output values
     */
//    println("output data X : ")
//    network.hiddenLayer.perceptrons.foreach(p => println( p.id + " - " + p.output + " ; bias : " + network.getBias(p.id.replace("perceptron", "bias")).output))
//    println("output data X Tilt : ")
//    listOfOutputXTilt.foreach(p => println(p.id + " - " + p.output + "; bias : "+ network.getBias(p.id.replace("perceptron", "bias")).output))
//    println()

    /**
     * update weight of synapsies
     */
    var information = "\n\nget update weight of synapsies \n f(x) = W + alpha ((h(x)*x) + (h(xtilt)*xtilt))\n"
    network.hiddenLayer.perceptrons.foreach { perceptron =>

      val hX = listOfOutputX.find(x => x.id.equals(perceptron.id)).get.output

      val hXTilt = listOfOutputXTilt.find(x => x.id.equals(perceptron.id)).get.output

//      println("X-tilt: " + hXTilt + " --> hX: " + hX+" | next | ")

      network.getSynapsiesTo(perceptron.id).foreach { synapsys =>

        /**
         * print information
         */
        information += " to " + perceptron.id + " :> sysnapsis: "+synapsys.to.id+" -> "+ synapsys.from.id+ "\n"

        information += "f(x) = "+synapsys.weight + " + (" +parameter.learningRate +" * ((" +hX+" * "+dataX.find( p => p.id.equals(synapsys.from.id)).get.output+
          ")"+"-"+"("+hXTilt+" * "+dataXTilt.find( p => p.id.equals(synapsys.from.id)).get.sample+")))"

        /**
         * main computation for weigh update
         */
        synapsys.deltaWeight = parameter.learningRate * ((hX *
          dataX.find( p => p.id.equals(synapsys.from.id)).get.output)
          - (hXTilt * dataXTilt.find( p => p.id.equals(synapsys.from.id)).get.sample))

        synapsys.weight = synapsys.weight + synapsys.deltaWeight

        /**
         * result information
         */
        information += " = "+synapsys.weight+"\n"
      }
    }

    /**
     * print information of synapsies weight update
     */
//    println(information)
//    println()

    /**
     * set information for update bias of hidden layer step
     */
    information = "b + alpha*(h(x) - h(xtilt))\n"
    /**
     * update biases hidden layer (bi)
     */
    network.hiddenLayer.biases.foreach { p =>

      /**
       * biases update of hidden layer information
       */
      information += "to"+p.id+"\n"
      information += p.output + " + (" + parameter.learningRate +"*(" +listOfOutputX.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.sample+
        " - "+listOfOutputXTilt.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.output+"))"

      /**
       * main hidden layer bias update computation
       */
      p.output = p.output + (parameter.learningRate *
        listOfOutputX.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.sample -
        listOfOutputXTilt.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.output)
      information += " = " + p.output + " \n"
    }

    /**
     * print information for biases update
     */
//    println(information)
//    println()

    information = "f(x) = b + alpha*(x - xtilt)\n"
    /**
     * update biases input layer (ci)
     */
    network.inputLayer.biases.foreach { p =>

      /**
       * biases for visible layer information
       */
      information += "to"+p.id+"\n"
      information += p.output + " + (" + parameter.learningRate +"*(" +dataX.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.output+
        " - "+dataXTilt.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.sample+"))\n"

      /**
       * main visible biases update computation
       */
      p.output = p.output + (dataX.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.output -
        dataXTilt.find(x => x.id.equals(p.id.replace("bias","perceptron"))).get.sample)

      information += " = " + p.output + " \n"
    }

    /**
     * print information for biases update of input layer
     */
//    println(information)
//    println()

    /**
     * get exp of hidden layer weight
     */
    val tempExpOutput = network.hiddenLayer.perceptrons.map { p =>
//        println(p.id + " : " + math.exp(network.getPerceptronWeightTo(p, false)) + "\n")
        (p.id, math.exp(network.getPerceptronWeightTo(p, false)))
    } toMap

      /**
       * set information for Free Energy Function
       */

    information = "cx + sigma(log(1+exp(bj+Wjx))\n f(x) = "
    network.inputLayer.perceptrons.foreach( p => information += p.output +" * " + 
      network.getBias(p.id.replace("perceptron", "bias")).output + " + ")


    network.hiddenLayer.perceptrons.foreach( p => information += "log( 1 + "+tempExpOutput.get(p.id).get +" )+")

    /**
     * Free Energy function
     */
    val freeEnergy = Math.exp(network.inputLayer.perceptrons.map( p => p.output *
      network.getBias(p.id.replace("perceptron", "bias")).output).foldLeft(0.0D)((temp, value) => temp + value) +
      network.hiddenLayer.perceptrons.map(p => Math.log(1 + tempExpOutput.get(p.id).get )).foldLeft(0.0D)((temp, value) => temp + value))

    information += " = "+freeEnergy +" \n"
    information += "#" + network.inputLayer.perceptrons.map( p => p.output *
      network.getBias(p.id.replace("perceptron", "bias")).output).foldLeft(0.0D)((temp, value) => temp + value) + "#\n"

    information += "#"+ network.hiddenLayer.perceptrons.map(p => {
//      println("--->"+p.id+" : "+tempExpOutput.get(p.id).get)
      Math.log(1 + tempExpOutput.get(p.id).get) }).foldLeft(0.0D)((temp, value) => temp + value) + "#\n\n"

    /**
     * print last synapsis condition
     */
//    println("\n\nlast synapsies condition : ")
//    network.synapsies.foreach( s => println(s.from.id+"("+s.from.output+") - "+ s.to.id+"("+s.to.output+") = " + s.weight))
//    println("----------------------------------------------------------------------------------------------------------------")

    /**
     * print free energy information and result
     */
//    println(information)
//    println("free energy : " + freeEnergy)
//    println("----------------------------------------------")

    /**
     * return free energy and X_tilst sample for next epoch
     */
    (freeEnergy, dataXTilt)
  }
}