package com.deepwissen.ml.serialization

import java.io.{FileInputStream, File, FileOutputStream, StringWriter}

import com.deepwissen.ml.algorithm.{MarkovChain, RandomSynapsysFactory, Network}
import org.apache.commons.io.output.WriterOutputStream
import org.scalatest.FunSuite

/**
 * Created by hendri_k on 7/13/15.
 */
class MarkovChainSerialization$Test extends FunSuite{
  test("save model markov chain") {

    val network = MarkovChain(7,5, RandomSynapsysFactory())
    val writer = new StringWriter()
    val outputStream = new WriterOutputStream(writer)

    NetworkSerialization.save(network, outputStream)

    NetworkSerialization.save(network, new FileOutputStream(
      new File("target" + File.separator + "markov-chain-model.json")))


    val json = writer.toString
    println(json)
  }

  test("load model markov chain") {

    //    val inputStream = getClass.getResourceAsStream("target" + File.separator + "network-model.json")
    val network = NetworkSerialization.load(inputStream = new FileInputStream(
      new File("target" + File.separator + "markov-chain-model.json")), typeOfInference = "MarkovChain").asInstanceOf[MarkovChain]

    println("converter model network to network object")
    network.inputLayer.perceptrons.foreach { perceptron =>
      println(network.inputLayer.id + " input layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.inputLayer.id != null)
    }

    network.inputLayer.biases.foreach { perceptron =>
      println(network.inputLayer.id + " input biases layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.inputLayer.id != null)
    }

    println(network.inputLayer.id + " bias => " + network.inputLayer.bias.get.id)
    assert(network.inputLayer.bias.isDefined)
    assert(network.inputLayer.perceptrons.length == 7)
    assert(network.inputLayer.prev.isEmpty)
    assert(network.inputLayer.next.isDefined)
    assert(network.inputLayer.biases.length == 7)

    network.hiddenLayer.perceptrons.foreach { perceptron =>
      println(network.hiddenLayer.id + " output layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.hiddenLayer.id != null)
    }

    network.hiddenLayer.biases.foreach { perceptron =>
      println(network.hiddenLayer.id + " output layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.hiddenLayer.id != null)
    }

    assert(network.hiddenLayer.bias.isDefined)
    println(network.inputLayer.id + " bias => " + network.inputLayer.bias.get.id)
    assert(network.hiddenLayer.perceptrons.length == 5)
    assert(network.hiddenLayer.prev.isDefined)
    assert(network.hiddenLayer.next.isEmpty)
    assert(network.hiddenLayer.biases.length == 5)

    network.synapsies.foreach { synapsys =>
      println(s"synapsys from ${synapsys.from} to ${synapsys.to} with weight ${synapsys.weight}")
      assert(synapsys.from != null)
      assert(synapsys.to != null)
    }
    println("total synapsys => " + network.synapsies.length)
    assert(network.synapsies.length == 35)
  }
}
