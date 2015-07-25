/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.algorithm.networks.Network
import com.deepwissen.ml.utils.{Denomination, ContValue}
import redis.clients.jedis.Jedis

import scala.annotation.tailrec

/**
 * @author Eko Khannedy
 * @since 6/4/15
 */
object RedisBackpropagation extends AbstractBackpropagation[RedisDataset] {

  /**
   * Train implementation
   * @param network network
   * @param dataset dataset
   * @param parameter training parameter
   */
  override def doTrain(network: Network, dataset: RedisDataset, parameter: BackpropragationParameter): Unit = {

    @tailrec
    def iterate(iteration: Int, error: Double): Unit = {
      if (error < parameter.epsilon || iteration > parameter.iteration) {
        // stop iteration
      } else {
        // print information
        println(s"###### error : $error : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")

        // run training
        @tailrec
        def train(index: Long, error: Double): Double = {
          if (index >= dataset.length) error
          else {
            //need adjust for target class
            val data = dataset.redis.get(index.toString).split(",").map(x => ContValue(x.toDouble)).asInstanceOf[Array[Denomination[_]]]
            val trainError = doTrainData(data, network, parameter)
            train(index + 1, error + trainError)
          }
        }

        val trainError = train(0, 0.0)

        // next iteration
        iterate(iteration + 1, trainError / dataset.length)
      }
    }

    // start first iteration with given max error
    iterate(1, Double.MaxValue)

  }

}

case class RedisDataset(redis: Jedis,
                        length: Long)