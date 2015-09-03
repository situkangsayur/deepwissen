package com.deepwissen.ml.serialization

import java.io.{FileInputStream, File, FileOutputStream, StringWriter}

import com.deepwissen.ml.algorithm.{AutoencoderParameter, DeepNetworkParameter, BackpropragationParameter, RandomSynapsysFactory}
import com.deepwissen.ml.algorithm.networks.{DeepNetwork, Network}
import com.deepwissen.ml.function.SigmoidFunction
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.utils.{Denomination, BinaryValue, ContValue}
import org.apache.commons.io.output.WriterOutputStream
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 7/25/15.
 */
class DeepNetworkSerialization$Test extends FunSuite{



  val outlook = Map(
    "sunny" -> ContValue(0.0),
    "overcast" -> ContValue(1.0),
    "rainy" -> ContValue(2.0)
  )

  val temperature = Map(
    "hot" -> ContValue(0.0),
    "mild" -> ContValue(1.0),
    "cool" -> ContValue(2.0)
  )

  val humidity = Map(
    "high" -> ContValue(0.0),
    "normal" -> ContValue(1.0)
  )

  val windy = Map(
    "TRUE" -> ContValue(0.0),
    "FALSE" -> ContValue(1.0)
  )

  val play = Map(
    "no" -> BinaryValue(List(0.0,0.0)),
    "yes" -> BinaryValue(List(0.0,1.0))
  )

  val priorKnowledge: List[Map[String, Denomination[_]]] = List(outlook, temperature, humidity, windy, play)
  val simplePriorKnowledge: List[Map[String, Denomination[_]]] = List(outlook, temperature, humidity, play)

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

  val stringsSimple =
    """
      |sunny,hot,high,no
      |sunny,hot,high,no
      |overcast,hot,high,yes
      |rainy,mild,high,yes
      |rainy,cool,normal,yes
      |rainy,cool,normal,no
      |overcast,cool,normal,yes
      |sunny,mild,high,no
      |sunny,cool,normal,yes
      |rainy,mild,normal,yes
      |sunny,mild,normal,yes
      |overcast,mild,high,yes
      |overcast,hot,normal,yes
      |rainy,mild,high,no
    """.stripMargin.trim.split("\n")

  val dataset = strings.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  val simpleDataset = stringsSimple.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  /**
   * Training Parameter
   */
  val parameter = DeepNetworkParameter(
    hiddenLayerSize = List(3,3,3),
    outputPerceptronSize = 1,
    targetClassPosition = -1,
    iteration = 100000,
    epsilon = 0.000001,
    momentum = 0.50,
    learningRate = 0.50,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1,
    autoecoderParam = AutoencoderParameter(
      iteration = 50000,
      epsilon = 0.00001,
      momentum = 0.50,
      learningRate = 0.30,
      synapsysFactory = RandomSynapsysFactory(),
      activationFunction = SigmoidFunction
    )
  )

  val targetClass = if(parameter.targetClassPosition == -1) dataset.head.length - 1 else parameter.targetClassPosition
  val simpleTargetClass = if(parameter.targetClassPosition == -1) simpleDataset.head.length - 1 else parameter.targetClassPosition


  val tempDatasetPlayTennis = dataset.map(data => {
    data.map { case (index, value) =>
      priorKnowledge(index)(value)
    }
  }).toList

  val finalDataSet = StandardNormalization.normalize(
    tempDatasetPlayTennis, tempDatasetPlayTennis
    , targetClass)

  //  val labels

  finalDataSet.foreach { array =>
    println(array.mkString(","))
  }

  val tempDatasetPlayTennisSimple = simpleDataset.map(data => {
    data.map { case (index, value) =>
      simplePriorKnowledge(index)(value)
    }
  }).toList

  val simpleFinalDataSet = StandardNormalization.normalize(
    tempDatasetPlayTennisSimple, tempDatasetPlayTennisSimple
    , simpleTargetClass)

  //  val labels

//  simpleFinalDataSet.foreach { array =>
//    println(array.mkString(","))
//  }

  var logger  = LoggerFactory.getLogger("Main Objects")


  test("save model") {

    val network = DeepNetwork(parameter, finalDataSet)
//    val network = DeepNetwork(parameter, simpleFinalDataSet, synapsysFactory = RandomSynapsysFactory())
    val writer = new StringWriter()
    val outputStream = new WriterOutputStream(writer)

    NetworkSerialization.save(network, outputStream)

    NetworkSerialization.save(network, new FileOutputStream(
      new File("target" + File.separator + "network-model.json")))


    val json = writer.toString
    println(json)
  }

  test("load model") {

    val network = NetworkSerialization.load(inputStream = new FileInputStream(
      new File("target" + File.separator + "network-model.json")), typeOfInference = "DeepNet").asInstanceOf[DeepNetwork]
    println("converter model network to network object")
    network.inputLayer.perceptrons.foreach { perceptron =>
      println(network.inputLayer.id + " input layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.inputLayer.id != null)
    }
    println(network.inputLayer.id + " bias => " + network.inputLayer.bias.get.id)
    assert(network.inputLayer.bias.isDefined)
    assert(network.inputLayer.perceptrons.length == 4)
    assert(network.inputLayer.prev.isEmpty)
    assert(network.inputLayer.next.isDefined)

    network.hiddenLayers.foreach { layer =>
      layer.perceptrons.foreach { perceptron =>
        println(layer.id + " hidden layer => " + perceptron.id)
        assert(perceptron.id != null)
        assert(layer.id != null)
      }
      println(layer.id + " bias => " + layer.bias.get.id)
      assert(layer.bias.isDefined)
      assert(layer.perceptrons.length == 3)
      assert(layer.prev.isDefined)
      assert(layer.next.isDefined)
    }

    network.outputLayer.perceptrons.foreach { perceptron =>
      println(network.outputLayer.id + " output layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.outputLayer.id != null)
    }
    assert(network.outputLayer.bias.isEmpty)
    assert(network.outputLayer.perceptrons.length == 1)
    assert(network.outputLayer.prev.isDefined)
    assert(network.outputLayer.next.isEmpty)

    network.synapsies.foreach { synapsys =>
      println(s"synapsys from ${synapsys.from} to ${synapsys.to} with weight ${synapsys.weight}")
      assert(synapsys.from != null)
      assert(synapsys.to != null)
    }
    println("total synapsys => " + network.synapsies.length)
    assert(network.synapsies.length == 43)
  }


}
