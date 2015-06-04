/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import java.io.{File, FileInputStream, FileOutputStream}

import com.deepwissen.ml.function.SigmoidFunction
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import org.scalatest.FunSuite
import redis.clients.jedis.Jedis

/**
 * @author Eko Khannedy
 * @since 6/4/15
 */
class RedisBackpropagation$Test extends FunSuite {

  val outlook = Map(
    "sunny" -> 0.0,
    "overcast" -> 1.0,
    "rainy" -> 2.0
  )

  val temperature = Map(
    "hot" -> 0.0,
    "mild" -> 1.0,
    "cool" -> 2.0
  )

  val humidity = Map(
    "high" -> 0.0,
    "normal" -> 1.0
  )

  val windy = Map(
    "TRUE" -> 0.0,
    "FALSE" -> 1.0
  )

  val play = Map(
    "no" -> 0.0,
    "yes" -> 1.0
  )

  val priorKnowledge = List(outlook, temperature, humidity, windy, play)

  val strings =
    """
      |sunny,hot,high,FALSE,no
      |sunny,hot,high,TRUE,no
      |overcast,hot,high,FALSE,yes
      |rainy,mild,high,FALSE,yes
      |rainy,cool,normal,FALSE,yes
      |rainy,cool,normal,TRUE,no
      |overcast,cool,normal,TRUE,yes
      |sunny,mild,high,FALSE,no
      |sunny,cool,normal,FALSE,yes
      |rainy,mild,normal,FALSE,yes
      |sunny,mild,normal,TRUE,yes
      |overcast,mild,high,TRUE,yes
      |overcast,hot,normal,FALSE,yes
      |rainy,mild,high,TRUE,no
    """.stripMargin.trim.split("\n")

  val dataset = strings.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  val finalDataSet = StandardNormalization.normalize {
    dataset.map { data =>
      data.map { case (index, value) =>
        priorKnowledge(index)(value)
      }
    }.toList
  }

  val redis = new Jedis("localhost", 6379)
  val redisDataset = RedisDataset(redis, finalDataSet.length)

  finalDataSet.zipWithIndex.foreach {
    case (array, index) =>
      redis.set(index.toString, array.mkString(","))
  }

  /**
   * Training Parameter
   */
  val parameter = BackpropragationParameter(
    hiddenLayerSize = 1,
    iteration = 70000,
    epsilon = 0.000000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  test("traininig and classification and save model") {
    // training
    val network = RedisBackpropagation.train(redisDataset, parameter)

    // classification
    finalDataSet.foreach { data =>
      val realScore = BasicClassification(data, network, SigmoidFunction)
      val percent = Math.round(realScore * 100)
      val score = if (realScore > 0.7) 1.0 else 0.0
      println(s"real $realScore == percent $percent% == score $score == targetClass ${data(4)}")
      assert(score == data(4))
    }

    // save model
    NetworkSerialization.save(network, new FileOutputStream(
      new File("target" + File.separator + "cuaca.json")))
  }

  test("load model and classification") {

    // load model
    val network = NetworkSerialization.load(new FileInputStream(
      new File("target" + File.separator + "cuaca.json")))

    // classification
    finalDataSet.foreach { data =>
      val realScore = BasicClassification(data, network, SigmoidFunction)
      val percent = Math.round(realScore * 100)
      val score = if (realScore > 0.7) 1.0 else 0.0
      println(s"real $realScore == percent $percent% == score $score == targetClass ${data(4)}")
      assert(score == data(4))
    }
  }

}
