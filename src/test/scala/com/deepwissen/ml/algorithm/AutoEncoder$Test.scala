package com.deepwissen.ml.algorithm

import java.io.{FileInputStream, File, FileOutputStream}

import com.deepwissen.ml.function.{RangeThresholdFunction, EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{Denomination, BinaryValue, ContValue}
import com.deepwissen.ml.validation.{AutoencoderValidation, BackProValidation, SplitValidation, Validation}
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 6/21/15.
 */
class AutoEncoder$Test extends FunSuite{

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
  val parameter = BackpropragationParameter(
    hiddenLayerSize = 1,
    outputPerceptronSize = 2,
    targetClassPosition = -1,
    iteration = 100000,
    epsilon = 0.000001,
    momentum = 0.50,
    learningRate = 0.50,
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

  //  val labels

  finalDataSet.foreach { array =>
    println(array.mkString(","))
  }

  var logger  = LoggerFactory.getLogger("Main Objects")



  test("traininig and classification and save model") {
    // training
    val network = Autoencoder.train(finalDataSet, parameter)

    val validator = AutoencoderValidation()
    val result = validator.classification(network, BasicClassification, finalDataSet, SigmoidFunction)
    println(result)

    val validateResult = validator.validate(result, finalDataSet, targetClass)
    val accuration = validator.accuration(validateResult) {
      RangeThresholdFunction(0.15)
    }

    println("accuration : "+accuration)

    val threshold = RangeThresholdFunction(0.15)

    var trueCounter = 0
    var allData = 0

    // classification
    finalDataSet.foreach { data =>
      val realScore = BasicClassification(data, network, SigmoidFunction)
      realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
        val originalClass = data(p._2).asInstanceOf[ContValue].get
        val result = p._1
        val compare = threshold.compare(p._1, originalClass)
        println(s"real $p == score $compare == targetClass ${originalClass}")
        trueCounter = if(compare) trueCounter + 1 else trueCounter
        allData += 1
      })
      println("------------------------------------------------------------")
    }

    val percent = trueCounter * (100.0 / allData)

    println("result comparation : " + trueCounter + " :> in percent : " + percent)

    assert(percent >= 80)


    // save model
    NetworkSerialization.save(network, new FileOutputStream(
      new File("target" + File.separator + "cuaca.json")))
  }
//
//  test("load model and classifications") {
//
//    // load model
//    val network = NetworkSerialization.load(new FileInputStream(
//      new File("target" + File.separator + "cuaca.json")))
//
//    // classification
//    finalDataSet.foreach { data =>
//      val realScore = BasicClassification(data, network, SigmoidFunction)
//      val percent = Math.round(realScore * 100)
//      val score = if (realScore > 0.7) 1.0 else 0.0
//      println(s"real $realScore == percent $percent% == score $score == targetClass ${data(4)}")
//      assert(score == data(4))
//    }
//  }
//
//  test("split validation") {
//
//    val (trainDataSet, classificationDataSet) = SplitValidation.split(finalDataSet, 70 -> 30)
//    val network = BasicBackpropagation.train(trainDataSet, parameter)
//
//    val result = Validation.classification(network, BasicClassification, classificationDataSet, SigmoidFunction)
//    println(result)
//
//    val validateResult = Validation.validate(result, classificationDataSet, 4)
//    println(validateResult)
//    val accuration = Validation.accuration(validateResult) {
//      EitherThresholdFunction(0.7, 0.0, 1.0)
//    }
//
//    println(accuration)
//  }

}
