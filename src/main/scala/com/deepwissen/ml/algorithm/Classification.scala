package com.deepwissen.ml.algorithm

import com.deepwissen.ml.function.ActivationFunction

import scala.concurrent.{ExecutionContext, Future}

/**
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
  def apply(data: DATA, model: MODEL, activationFunction: ActivationFunction): Double

  /**
   * Run classification async
   * @param data data
   * @param model model
   * @param activationFunction activation function
   * @param executionContext execution context
   * @return
   */
  def async(data: DATA, model: MODEL, activationFunction: ActivationFunction)
           (implicit executionContext: ExecutionContext): Future[Double] =
    Future(apply(data, model, activationFunction))

}

/**
 * Basic implementation of classification for data array of double and network model
 */
object BasicClassification extends Classification[Array[Double], Network] {

  /**
   * Run classification
   * @param data data
   * @param network model
   * @param activationFunction activation function
   * @return classification result
   */
  override def apply(data: Array[Double], network: Network, activationFunction: ActivationFunction): Double = {

    // fill input layer
    network.inputLayer.fillOutput(data)

    // fill hidden layer
    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        perceptron.weight = network.getPerceptronWeight(perceptron)
        perceptron.output = activationFunction.activation(perceptron.weight)
      }
    }

    // fill output layer
    network.outputLayer.perceptrons.foreach { perceptron =>
      perceptron.weight = network.getPerceptronWeight(perceptron)
      perceptron.output = activationFunction.activation(perceptron.weight)
    }

    // calculate result
    network.outputLayer.perceptrons.foldLeft(0.0) { (value, perceptron) =>
      value + perceptron.output
    }
  }
}