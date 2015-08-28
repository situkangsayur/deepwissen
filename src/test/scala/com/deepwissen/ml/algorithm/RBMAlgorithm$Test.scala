package com.deepwissen.ml.algorithm

import java.io.{File, FileOutputStream}

import com.deepwissen.ml.function.{RangeThresholdFunction, EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{Denomination, BinaryValue, ContValue}
import com.deepwissen.ml.validation.{MarkovChainValidation, Validation}
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 7/18/15.
 */
class RBMAlgorithm$Test extends FunSuite {


//  val outlook = Map(
//    "sunny" -> ContValue(0.0),
//    "overcast" -> ContValue(1.0),
//    "rainy" -> ContValue(2.0)
//  )
//
//  val temperature = Map(
//    "hot" -> ContValue(0.0),
//    "mild" -> ContValue(1.0),
//    "cool" -> ContValue(2.0)
//  )
//
//  val humidity = Map(
//    "high" -> ContValue(0.0),
//    "normal" -> ContValue(1.0)
//  )
//
//  val windy = Map(
//    "TRUE" -> ContValue(0.0),
//    "FALSE" -> ContValue(1.0)
//  )

  val outlook = Map(
    "no" -> ContValue(0.0),
    "yes" -> ContValue(1.0)
  )

  val temperature = Map(
    "no" -> ContValue(0.0),
    "yes" -> ContValue(1.0)
  )

  val humidity = Map(
    "no" -> ContValue(0.0),
    "yes" -> ContValue(1.0)
  )

  val windy = Map(
    "no" -> ContValue(0.0),
    "yes" -> ContValue(1.0)
  )


  val field6 = Map(
    "no" -> ContValue(0.0),
    "yes" -> ContValue(1.0)
  )

  val field5 = Map(
    "no" -> ContValue(0.0),
    "yes" -> ContValue(1.0)
  )

  val play = Map(
    "no" -> BinaryValue(List(0.0,0.0)),
    "yes" -> BinaryValue(List(0.0,1.0))
  )


  val priorKnowledge: List[Map[String, Denomination[_]]] = List(outlook, temperature, windy,humidity,field5,field6, play)
//  val priorKnowledge: List[Map[String, Denomination[_]]] = List(outlook, temperature, humidity,windy, play)

//  val strings =
//    """
//      |sunny,hot,high,FALSE,no
//      |sunny,hot,high,TRUE,no
//      |overcast,hot,high,FALSE,yes
//      |rainy,mild,high,FALSE,yes
//      |rainy,cool,normal,FALSE,yes
//      |rainy,cool,normal,TRUE,no
//      |overcast,cool,normal,TRUE,yes
//      |sunny,mild,high,FALSE,no
//      |sunny,cool,normal,FALSE,yes
//      |rainy,mild,normal,FALSE,yes
//      |sunny,mild,normal,TRUE,yes
//      |overcast,mild,high,TRUE,yes
//      |overcast,hot,normal,FALSE,yes
//      |rainy,mild,high,TRUE,no
//    """.stripMargin.trim.split("\n")

//  Array(1, 1, 1, 0, 0, 0),
//  Array(1, 0, 1, 0, 0, 0),
//  Array(1, 1, 1, 0, 0, 0),
//  Array(0, 0, 1, 1, 1, 0),
//  Array(0, 0, 1, 0, 1, 0),
//  Array(0, 0, 1, 1, 1, 0)

  val strings =
    """
      |yes,yes,yes,no,no,no,no
      |yes,no,yes,no,no,no,no
      |yes,yes,yes,no,no,no,no
      |no,no,yes,yes,yes,no,no
      |no,no,yes,no,yes,no,no
      |no,no,yes,yes,yes,no,no
    """.stripMargin.trim.split("\n")

  val stringsTest =
    """
      |yes,yes,no,no,no,no,no
      |no,no,no,yes,yes,no,no
    """.stripMargin.trim.split("\n")



//  val stringsTest =  """
//                      |sunny,hot,high,FALSE,no
//                      |sunny,hot,high,TRUE,no
//                      |overcast,hot,high,FALSE,yes
//                      |rainy,mild,high,FALSE,yes
//                      |rainy,cool,normal,FALSE,yes
//                      |rainy,cool,normal,TRUE,no
//                      |overcast,cool,normal,TRUE,yes
//                      |sunny,mild,high,FALSE,no
//                      |sunny,cool,normal,FALSE,yes
//                      |rainy,mild,normal,FALSE,yes
//                      |sunny,mild,normal,TRUE,yes
//                      |overcast,mild,high,TRUE,yes
//                      |overcast,hot,normal,FALSE,yes
//                      |rainy,mild,high,TRUE,no
//                    """.stripMargin.trim.split("\n")


  val dataset = strings.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  val datasetTest = stringsTest.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  /**
   * Training Parameter
   */
  val parameter = GibbsParameter(
    inputPerceptronSize = dataset.head.length - 1,
    hiddenPerceptronSize = 3,
    k = 100,
    iteration = 1000,
    epsilon = 0.00001,
    momentum = 0.50,
    learningRate = 0.2,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    dataSize = dataset.size
  )

  val targetClass = if(parameter.targetClassPosition == -1) dataset.head.length - 1 else parameter.targetClassPosition

  val finalDataSet = StandardNormalization.normalize(
    dataset.map(data => {
      data.map { case (index, value) =>{
        priorKnowledge(index)(value)
      }
      }
    }).toList
    , targetClass)


  val finalDataSetTest = StandardNormalization.normalize(
    datasetTest.map(data => {
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


  test("Testing for RBM Algorithm") {

    val tempNetwork = RBMAlgorithm.train(finalDataSet, parameter)

    val validator = MarkovChainValidation()
    val result = validator.classification(tempNetwork, RBMClassificationTesting, finalDataSetTest, SigmoidFunction)
    println(result)

    val validateResult = validator.validate(result, finalDataSetTest, targetClass)
    val accuration = validator.accuration(validateResult) {
      RangeThresholdFunction(0.15)
    }

    println("accuration : "+accuration)

//    assert(accuration >= 80)

    val threshold = RangeThresholdFunction(0.15)

    var trueCounter = 0
    var allData = 0

    // classification
    finalDataSetTest.foreach { data =>
      val realScore = RBMClassificationTesting(data, tempNetwork, SigmoidFunction)
      realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
        val originalClass = data(p._2).asInstanceOf[ContValue].get
        val result = p._1
        val compare = threshold.compare(p._1, originalClass)
        println(s"real $p == score $compare == targetClass ${originalClass}")
        trueCounter = if(compare._1) trueCounter + 1 else trueCounter
        allData += 1
      })
      println("------------------------------------------------------------")
    }

    val percent = trueCounter * (100.0 / allData)

    println("result comparation : " + trueCounter + " :> in percent : " + percent)

    assert(percent >= 80)


    // save model
    NetworkSerialization.save(tempNetwork, new FileOutputStream(
      new File("target" + File.separator + "cuaca.json")))
  }


}
