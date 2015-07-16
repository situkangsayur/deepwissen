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
    println(network.inputLayer.id + " bias => " + network.inputLayer.bias.get.id)
    assert(network.inputLayer.bias.isDefined)
    assert(network.inputLayer.perceptrons.length == 7)
    assert(network.inputLayer.prev.isEmpty)
    assert(network.inputLayer.next.isDefined)

//    network.hiddenLayers.foreach { layer =>
//      layer.perceptrons.foreach { perceptron =>
//        println(layer.id + " hidden layer => " + perceptron.id)
//        assert(perceptron.id != null)
//        assert(layer.id != null)
//      }
//      println(layer.id + " bias => " + layer.bias.get.id)
//      assert(layer.bias.isDefined)
//      assert(layer.perceptrons.length == 2)
//      assert(layer.prev.isDefined)
//      assert(layer.next.isDefined)
//    }

    network.outputLayer.perceptrons.foreach { perceptron =>
      println(network.outputLayer.id + " output layer => " + perceptron.id)
      assert(perceptron.id != null)
      assert(network.outputLayer.id != null)
    }
    assert(network.outputLayer.bias.isEmpty)
    assert(network.outputLayer.perceptrons.length == 5)
    assert(network.outputLayer.prev.isDefined)
    assert(network.outputLayer.next.isEmpty)

    network.synapsies.foreach { synapsys =>
      println(s"synapsys from ${synapsys.from} to ${synapsys.to} with weight ${synapsys.weight}")
      assert(synapsys.from != null)
      assert(synapsys.to != null)
    }
    println("total synapsys => " + network.synapsies.length)
    assert(network.synapsies.length == 40)
  }
}
