/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import java.io.{File, FileInputStream, FileOutputStream}
import java.util.UUID

import com.deepwissen.ml.function.SigmoidFunction
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import org.apache.hadoop.hbase._
import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.util.Bytes
import org.scalatest.FunSuite

/**
 * @author Eko Khannedy
 * @since 6/4/15
 */
class HBaseBackpropagation$Test extends FunSuite {

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

  val config = HBaseConfiguration.create()
  config.set("hbase.zookeeper.quorum", "localhost")
  config.set("hbase.zookeeper.property.clientPort", "2181")
  val connection = ConnectionFactory.createConnection(config)

  if (connection.getAdmin.getNamespaceDescriptor("backpropagation") == null) {
    connection.getAdmin.createNamespace(NamespaceDescriptor.create("backpropagation").build())
  }

  if (!connection.getAdmin.isTableAvailable(TableName.valueOf("backpropagation:training_data"))) {
    connection.getAdmin.createTable(
      new HTableDescriptor(TableName.valueOf("backpropagation:training_data")).
        addFamily(new HColumnDescriptor("dataset"))
    )
    val table = connection.getTable(TableName.valueOf("backpropagation:training_data"))
    finalDataSet.foreach { array =>
      val put = new Put(Bytes.toBytes(UUID.randomUUID().toString))
        .addColumn(Bytes.toBytes("dataset"), Bytes.toBytes("0"), Bytes.toBytes(array(0)))
        .addColumn(Bytes.toBytes("dataset"), Bytes.toBytes("1"), Bytes.toBytes(array(1)))
        .addColumn(Bytes.toBytes("dataset"), Bytes.toBytes("2"), Bytes.toBytes(array(2)))
        .addColumn(Bytes.toBytes("dataset"), Bytes.toBytes("3"), Bytes.toBytes(array(3)))
        .addColumn(Bytes.toBytes("dataset"), Bytes.toBytes("4"), Bytes.toBytes(array(4)))

      table.put(put)
    }
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

  val table = connection.getTable(TableName.valueOf("backpropagation:training_data"))

  val hbaseDataset = HBaseDataset(
    resultScanner = () => table.getScanner(new Scan().setCaching(100).setBatch(100)),
    length = finalDataSet.length,
    converter = (result: Result) => Array(
      Bytes.toDouble(result.getValue(Bytes.toBytes("dataset"), Bytes.toBytes("0"))),
      Bytes.toDouble(result.getValue(Bytes.toBytes("dataset"), Bytes.toBytes("1"))),
      Bytes.toDouble(result.getValue(Bytes.toBytes("dataset"), Bytes.toBytes("2"))),
      Bytes.toDouble(result.getValue(Bytes.toBytes("dataset"), Bytes.toBytes("3"))),
      Bytes.toDouble(result.getValue(Bytes.toBytes("dataset"), Bytes.toBytes("4")))
    )
  )

  test("traininig and classification and save model") {
    // training
    val network = HBaseBackpropagation.train(hbaseDataset, parameter)

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
