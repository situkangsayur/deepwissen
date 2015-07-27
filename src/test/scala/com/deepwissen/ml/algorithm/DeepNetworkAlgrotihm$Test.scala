package com.deepwissen.ml.algorithm

import java.io.{FileInputStream, File, FileOutputStream}

import com.deepwissen.ml.algorithm.networks.Network
import com.deepwissen.ml.function.{EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{Denomination, BinaryValue, ContValue}
import com.deepwissen.ml.validation.{DeepNetworkValidation, BackProValidation}
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 7/25/15.
 */
class DeepNetworkAlgrotihm$Test extends FunSuite{

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

  /**
   * Training Parameter
   */
  val parameter = DeepNetworkParameter(
    hiddenLayerSize = List(3,3,3),
    outputPerceptronSize = 2,
    targetClassPosition = -1,
    iteration = 70000,
    epsilon = 0.000000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  val targetClass = if(parameter.targetClassPosition == -1) dataset.head.length - 1 else parameter.targetClassPosition

  val finalDataSet = StandardNormalization.normalize(
    dataset.map(data => {
      data.map { case (index, value) =>
        priorKnowledge(index)(value)
      }
    }).toList
    , targetClass)
  3
  finalDataSet.foreach { array =>
    println(array.mkString(","))
  }


  var logger  = LoggerFactory.getLogger("Main Objects")

  test("traininig and classification and save model") {
    // training
    try {

      logger.info(finalDataSet.toString())

      val network = DeepNetworkAlgorithm.train(finalDataSet, parameter)

      val validator = DeepNetworkValidation()

      val result = validator.classification(network, DeepNetworkClassification, finalDataSet, SigmoidFunction)
      logger.info("result finding : "+ result.toString())

      val validateResult = validator.validate(result, finalDataSet, 4)

      logger.info("after validation result : "+validateResult.toString())

      val accuration = validator.accuration(validateResult) {
        EitherThresholdFunction(0.7, 0.0, 1.0)
      }

      logger.info("after accuration counting : "+accuration.toString())

      // classification
      finalDataSet.foreach { data =>
        val realScore = DeepNetworkClassification(data, network, SigmoidFunction)
        realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
          val percent = Math.round(p._1 * 100.0)
          val score = if (p._1 > 0.7) 1.0 else 0.0
          val originalClass = data(targetClass).asInstanceOf[BinaryValue].get(p._2)
          println(s"real $p== percent $percent% == score $score == targetClass ${originalClass}")
          assert(score == originalClass)
        })
      }

      // save model
      NetworkSerialization.save(network, new FileOutputStream(
        new File("target" + File.separator + "cuaca.json")))
    }catch {
      case npe : NullPointerException => npe.printStackTrace()
      case e : Exception => e.printStackTrace()
    }
  }

  test("load model and classification") {

    // load model
    val network = NetworkSerialization.load(inputStream = new FileInputStream(
      new File("target" + File.separator + "cuaca.json")), typeOfInference = "NeuralNet").asInstanceOf[Network]

    // classification
    finalDataSet.foreach { data =>
      val realScore = BasicClassification(data, network, SigmoidFunction)
      realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
        val percent = Math.round(p._1 * 100.0)
        val score = if (p._1 > 0.7) 1.0 else 0.0
        val originalClass = data(targetClass).asInstanceOf[BinaryValue].get(p._2)
        println(s"real $p== percent $percent% == score $score == targetClass ${originalClass}")
        assert(score == originalClass)
      })
    }
  }

}
