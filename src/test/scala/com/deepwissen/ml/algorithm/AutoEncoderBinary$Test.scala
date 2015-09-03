package com.deepwissen.ml.algorithm

import java.io.{File, FileOutputStream}

import com.deepwissen.ml.function.{EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{Denomination, BinaryValue, ContValue}
import com.deepwissen.ml.validation.{AutoencoderValidation, BackProValidation, Validation}
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 6/30/15.
 */
class AutoEncoderBinary$Test extends FunSuite{
  val outlook = Map(
    "sunny" -> BinaryValue(List(0.0,0.0)),
    "overcast" -> BinaryValue(List(0.0,1.0)),
    "rainy" -> BinaryValue(List(1.0,0.0))
  )

  val temperature = Map(
    "hot" -> BinaryValue(List(0.0,0.0)),
    "mild" -> BinaryValue(List(0.0,1.0)),
    "cool" -> BinaryValue(List(1.0,0.0))
  )

  val humidity = Map(
    "high" -> BinaryValue(List(0.0,0.0)),
    "normal" -> BinaryValue(List(0.0,1.0))
  )

  val windy = Map(
    "TRUE" -> BinaryValue(List(0.0,0.0)),
    "FALSE" -> BinaryValue(List(0.0,1.0))
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
  val parameter = AutoencoderParameter(
    hiddenPerceptronSize = 1,
    iteration = 70000,
    epsilon = 0.000000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  val targetClass = -1

  val tempDataset = dataset.map(data => {
    data.map { case (index, value) =>
      priorKnowledge(index)(value)
    }
  }).toList

  val finalDataSet = StandardNormalization.normalize(
    tempDataset, tempDataset
    , targetClass)

  //  val labels

  finalDataSet.foreach { array =>
    println(array.mkString(","))
  }

  var logger  = LoggerFactory.getLogger("Main Objects")



  test("traininig and classification using binary and save model") {
    // training
    val network = Autoencoder.train(finalDataSet, parameter)


    val validator = AutoencoderValidation()
    val result = validator.classification(network, AutoencoderClassification, finalDataSet, SigmoidFunction)
    println(result)

    val validateResult = validator.validate(result, finalDataSet, 4)
    val accuration = validator.accuration(validateResult) {
      EitherThresholdFunction(0.7, 0.0, 1.0)
    }

    println("accuration : "+accuration._1 +": Recall : " + accuration._2 + " : precision : " + accuration._3)

    // classification
    finalDataSet.foreach { data =>
      val realScore = AutoencoderClassification(data, network, SigmoidFunction)
      realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
        val percent = Math.round(p._1 * 100.0)
        val score = if (p._1 > 1/3) 1.0 else 0.0
        val originalClass = data(p._2).asInstanceOf[ContValue].get
        println(s"real $p== percent $percent% == score $score == targetClass ${originalClass}")
        assert(score == originalClass)
      })
    }


    // save model
    NetworkSerialization.save(network, new FileOutputStream(
      new File("target" + File.separator + "cuaca.json")))
  }
}
