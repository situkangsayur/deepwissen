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
object BasicClassification extends Classification[Array[Denomination[_]], InferencesNetwork] {

  /**
   * Run classification
   * @param data data
   * @param network model
   * @param activationFunction activation function
   * @return classification result
   */
  override def apply(data: Array[Denomination[_]], network: InferencesNetwork, activationFunction: ActivationFunction): Denomination[_] = {

    val tempNetwork = network.asInstanceOf[Network]
    
    // fill input layer
    tempNetwork.inputLayer.fillOutput(data)

    // fill hidden layer
    tempNetwork.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        perceptron.weight = tempNetwork.getPerceptronWeightTo(perceptron)
        perceptron.output = activationFunction.activation(perceptron.weight)
      }
    }

    // fill output layer
    tempNetwork.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = tempNetwork.getPerceptronWeightTo(perceptron)
      perceptron.output = activationFunction.activation(perceptron.weight)
    }

    BinaryValue(tempNetwork.outputLayer.perceptrons.map(x => x.output))
    // calculate result
//    network.outputLayer.perceptrons.foldLeft(0.0) { (value, perceptron) =>
//      value + perceptron.output
//    }
  }
}



/**
 * implementation of classification for data array of double and markov model for RBM
 * @author Hendri Karisma
 * @since 7/23/15
 */
object RBMClassificationTesting extends Classification[Array[Denomination[_]], InferencesNetwork] {

  /**
   * Run classification
   * @param data data
   * @param network model
   * @param activationFunction activation function
   * @return classification result
   */
  override def apply(data: Array[Denomination[_]], network: InferencesNetwork, activationFunction: ActivationFunction): Denomination[_] = {

    val tempNetwork = network.asInstanceOf[MarkovChain]
    // fill input layer
    tempNetwork.inputLayer.fillOutput(data)

    tempNetwork.inputLayer.perceptrons.foreach( p => println(p.id + ": " + p.output))
    println()
    tempNetwork.synapsies.foreach( s => {
      println("from : " + s.from.id + "( "+ s.from.output + " )"+ " - " + "to : " + s.to.id + "( " + s.to.output+ " ) = " + s.deltaWeight )
    })

    // fill hidden layer

    tempNetwork.hiddenLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = tempNetwork.getPerceptronWeightTo(perceptron)
      perceptron.output = activationFunction.activation(perceptron.weight)
    }


    // fill output layer
    tempNetwork.inputLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = tempNetwork.getPerceptronWeightTo(perceptron)
      perceptron.output = activationFunction.activation(perceptron.weight)
      print(perceptron.output+ " ; ")
    }
    println()

    BinaryValue(tempNetwork.inputLayer.perceptrons.map(x => x.output))
    // calculate result
    //    network.outputLayer.perceptrons.foldLeft(0.0) { (value, perceptron) =>
    //      value + perceptron.output
    //    }
  }
}
