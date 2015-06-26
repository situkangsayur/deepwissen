/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.utils.Denomination

import scala.annotation.tailrec

/**
 * Implementation of Neural Network Backpropagation
 * @author Eko Khannedy
 * @since 2/25/15
 */
object BasicBackpropagation extends AbstractBackpropagation[List[Array[Denomination[_]]]] {

  /**
   * Run training with given dataset
   * @param dataset dataset
   * @param parameter training parameter
   * @return network model
   */
  override def doTrain(network: Network, dataset: List[Array[Denomination[_]]], parameter: BackpropragationParameter): Unit = {

    @tailrec
    def iterate(iteration: Int, error: Double): Unit = {
      if (error < parameter.epsilon || iteration > parameter.iteration) {
        // stop iteration
      } else {
        // print information
        println(s"###### error : $error : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
        // run training
        val trainError = dataset.foldLeft(0.0)((value, data) => value + doTrainData(data, network, parameter))
        // next iteration
        iterate(iteration + 1, trainError / (2 * dataset.length * parameter.outputPerceptronSize))
      }
    }

    // start first iteration with given max error
    iterate(1, Double.MaxValue)
  }
}
