package com.deepwissen.ml.algorithm

import java.io.{FileInputStream, File, FileOutputStream}

import com.deepwissen.ml.function.{EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{Denomination, TargetValue, FieldValue}
import com.deepwissen.ml.validation.{SplitValidation, Validation}
import org.scalatest.FunSuite

/**
 * Created by hendri_k on 6/21/15.
 */
class AutoEncoder$Test extends FunSuite{

  val outlook = Map(
    "sunny" -> FieldValue(0.0),
    "overcast" -> FieldValue(1.0),
    "rainy" -> FieldValue(2.0)
  )

  val temperature = Map(
    "hot" -> FieldValue(0.0),
    "mild" -> FieldValue(1.0),
    "cool" -> FieldValue(2.0)
  )

  val humidity = Map(
    "high" -> FieldValue(0.0),
    "normal" -> FieldValue(1.0)
  )

  val windy = Map(
    "TRUE" -> FieldValue(0.0),
    "FALSE" -> FieldValue(1.0)
  )

  val play = Map(
    "no" -> TargetValue(List(0.0,1.0)),
    "yes" -> TargetValue(List(1.0, 0.0))
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


  /**
   * Training Parameter
   */
  val parameter = BackpropragationParameter(
    hiddenLayerSize = 1,
    outputPerceptronSize = 1,
    targetClassPosition = -1,
    iteration = 70000,
    epsilon = 0.000000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  val dataset = strings.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }
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


  test("traininig and classification and save model") {
    // training
    val network = Autoencoder.train(finalDataSet, parameter)

    val result = Validation.classification(network, BasicClassification, finalDataSet, SigmoidFunction)
    println(result)

    val validateResult = Validation.validate(result, finalDataSet, 4)
    val accuration = Validation.accuration(validateResult) {
      EitherThresholdFunction(0.7, 0.0, 1.0)
    }

    println(accuration)

    // classification
    finalDataSet.foreach { data =>
      val realScore = BasicClassification(data, network, SigmoidFunction)
      realScore.asInstanceOf[TargetValue].get.zipWithIndex.foreach(p => {
        val percent = Math.round(p._1 * 100.0)
        val score = if (p._1 > 0.7) 1.0 else 0.0
        val originalClass = data(targetClass).asInstanceOf[TargetValue].get(p._2)
        println(s"real $p== percent $percent% == score $score == targetClass ${originalClass}")
        assert(score == originalClass)
      })
    }

    // save model
    NetworkSerialization.save(network, new FileOutputStream(
      new File("target" + File.separator + "cuaca.json")))
  }
//
//  test("load model and classification") {
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
