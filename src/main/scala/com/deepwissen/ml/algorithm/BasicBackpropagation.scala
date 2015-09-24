/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import java.io._


import org.slf4j.{LoggerFactory, Logger}

import scala.io.Source
import com.deepwissen.ml.algorithm.networks.Network
import com.deepwissen.ml.utils.Denomination
import scala.annotation.tailrec


/**
 * Implementation of Neural Network Backpropagation
 * @author Eko Khannedy
 * @since 2/25/15
 */
object BasicBackpropagation extends AbstractBackpropagation[List[Array[Denomination[_]]]] {
//  val writer = new PrintWriter(new File("output-ds1-bp.txt"))
  var logger : Logger = LoggerFactory.getLogger(this.getClass)
  val file = new PrintStream(new FileOutputStream("result-ds1-bp1-2003.txt"), true)
//  val bw = new BufferedWriter(new FileWriter(file))

//  val writer : Output = Resource.fromFile("output-ds1-bp.txt")
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
//      if (error < parameter.epsilon) {
        // stop iteration
      } else {
        // print information

        val rmse = math.sqrt(error)
        if(iteration % 100 == 0){
//          logger.info(s"###### MSE : $error : RMSE : $rmse : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
          file.append(s"###### MSE : $error : RMSE : $rmse : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
        }
        println(s"###### MSE : $error : RMSE : $rmse : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
        // run training
        val trainError = dataset.foldLeft(0.0)((value, data) => value + doTrainData(data, network, parameter))
        // next iteration
        iterate(iteration + 1, trainError / (dataset.length * parameter.outputPerceptronSize))
      }
    }

    // start first iteration with given max error
    iterate(1, Double.MaxValue)
    file.close()

  }
}
