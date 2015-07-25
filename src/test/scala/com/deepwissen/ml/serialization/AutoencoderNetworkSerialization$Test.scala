package com.deepwissen.ml.serialization

import java.io.{FileInputStream, File, FileOutputStream, StringWriter}

import com.deepwissen.ml.algorithm.RandomSynapsysFactory
import com.deepwissen.ml.algorithm.networks.{AutoencoderNetwork, Network}
import org.apache.commons.io.output.WriterOutputStream
import org.scalatest.FunSuite

/**
 * Created by hendri_k on 7/26/15.
 */
class AutoencoderNetworkSerialization$Test extends FunSuite{

  test("save model") {

    val network = AutoencoderNetwork(inputPerceptronSize = 4, hiddenPerceptronSize =  3, synapsysFactory = RandomSynapsysFactory())
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
      new File("target" + File.separator + "network-model.json")), typeOfInference = "AutoencoderNet").asInstanceOf[AutoencoderNetwork]
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


      network.hiddenLayer.perceptrons.foreach { perceptron =>
        println(network.hiddenLayer.id + " hidden layer => " + perceptron.id)
        assert(perceptron.id != null)
        assert(network.hiddenLayer.id != null)
      }
      println(network.hiddenLayer.id + " bias => " + network.hiddenLayer.bias.get.id)
      assert(network.hiddenLayer.bias.isDefined)
      assert(network.hiddenLayer.perceptrons.length == 3)
      assert(network.hiddenLayer.prev.isDefined)
      assert(network.hiddenLayer.next.isDefined)


    network.outputLayer.perceptrons.foreach { perceptron =>
      println(network.outputLayer.id + " output layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.outputLayer.id != null)
    }
    assert(network.outputLayer.bias.isEmpty)
    assert(network.outputLayer.perceptrons.length == 4)
    assert(network.outputLayer.prev.isDefined)
    assert(network.outputLayer.next.isEmpty)

    network.synapsies.foreach { synapsys =>
      println(s"synapsys from ${synapsys.from} to ${synapsys.to} with weight ${synapsys.weight}")
      assert(synapsys.from != null)
      assert(synapsys.to != null)
    }
    println("total synapsys => " + network.synapsies.length)
    assert(network.synapsies.length == 31)
  }

}
