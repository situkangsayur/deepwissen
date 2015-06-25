/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.utils.Denomination
import org.apache.hadoop.hbase.client._

import scala.annotation.tailrec

/**
 * @author Eko Khannedy
 * @since 6/4/15
 */
object HBaseBackpropagation extends AbstractBackpropagation[HBaseDataset] {

  /**
   * Train implementation
   * @param network network
   * @param dataset dataset
   * @param parameter training parameter
   */
  override def doTrain(network: Network, dataset: HBaseDataset, parameter: BackpropragationParameter): Unit = {

    @tailrec
    def iterate(iteration: Int, error: Double): Unit = {
      if (error < parameter.epsilon || iteration > parameter.iteration) {
        // stop iteration
      } else {
        // print information
        println(s"###### error : $error : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")

        // get iterator from result scanner
        val iterator = dataset.resultScanner().iterator()

        // run training
        @tailrec
        def train(error: Double): Double = {
          if (iterator.hasNext) {
            val data = dataset.converter(iterator.next())
            val trainResult = doTrainData(data.asInstanceOf[Array[Denomination[_]]], network, parameter)
            train(error + trainResult)
          } else {
            error
          }
        }

        val trainError = train(0.0)

        // next iteration
        iterate(iteration + 1, trainError / dataset.length)
      }
    }

    // start first iteration with given max error
    iterate(1, Double.MaxValue)

  }

}

/**
 * HBase dataset for training
 * @param resultScanner result scanner
 * @param length length of dataset
 * @param converter converter for convert from Result to Array[Double]
 */
case class HBaseDataset(resultScanner: () => ResultScanner,
                        length: Long,
                        converter: Result => Array[Any])
