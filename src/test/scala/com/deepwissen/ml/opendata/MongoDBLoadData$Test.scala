package com.deepwissen.ml.opendata

import com.deepwissen.ml.algorithm.{BackpropragationParameter, RandomSynapsysFactory}
import com.deepwissen.ml.function.SigmoidFunction
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.utils.{BinaryValue, ContValue, Denomination}
import com.mongodb.casbah.Imports._
import com.mongodb.casbah.MongoClient
import com.mongodb.casbah.commons.MongoDBObject
import org.scalatest.FunSuite

import scala.xml.XML
import scalaz.BiConstrainedNaturalTransformation

/**
 * Created by hendri_k on 8/27/15.
 */
class MongoDBLoadData$Test extends FunSuite{

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
    "no" -> BinaryValue(List(0.0)),
    "yes" -> BinaryValue(List(1.0))
  )

  val priorKnowledge: List[Map[String, Denomination[_]]] = List(outlook, temperature, humidity, windy, play)
  val fieldNames: List[String] = List("outlook", "temperature", "humidity", "windy", "play")

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
    iteration = 70000,
    epsilon = 0.000000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  val targetClass = if(parameter.targetClassPosition == -1) dataset.head.length - 1 else parameter.targetClassPosition

  val tempDatasetPlayTennis = dataset.map(data => {
    data.map { case (index, value) =>
      priorKnowledge(index)(value)
    }
  }).toList

  val finalDataSet = StandardNormalization.normalize(
    tempDatasetPlayTennis, tempDatasetPlayTennis
    , targetClass)

  val mongoClient =  MongoClient()

  test("test for load database continue value"){
    // Connect to default

    val db = mongoClient("sample")
    val collection = db("playtennis")

    collection.drop()


    val result = finalDataSet.map { x =>
      collection.insert(MongoDBObject(x.zipWithIndex.map( p => {
        val temp = if(p._1.isInstanceOf[BinaryValue])
          ContValue(p._1.asInstanceOf[BinaryValue].get(0))
        else p._1.asInstanceOf[ContValue]

        (fieldNames(p._2) -> temp.get)
      }).toList))
    }

    val loadDataset = collection.find().map { p =>
      fieldNames.map(x => (ContValue(p.getAs[Double](x).get))).toArray
    }.toList

    loadDataset.foreach { p=>
      p.foreach( x => print(x.get + ", "))
      println()
    }

    assert(loadDataset.size == 14)
    assert(loadDataset(0).size == 5)
  }


  /**
   * Training Parameter
   */
  val parameterBank = BackpropragationParameter(
    hiddenLayerSize = 1,
    outputPerceptronSize = 2,
    targetClassPosition = -1,
    iteration = 1000,
    epsilon = 0.000000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )





  test("test for load database with descrete value"){

    val featuresName = List("ID_BANK","ID_LAPORAN1","NAMA_BANK","TAHUN","BULAN","Illiquid_Assets","Illiquid_Liabilities",
      "LTR","Giro","Tabungan","Deposito","DPK","CASA","CORE_DEPOSITS","Kredit","FINANCING_GAP","TOTAL_ASET","ATMR","RWA",
      "CAR","TotalEkuitas","EQTA","LABA_RUGI_TAHUN_BERJALAN","LABA_RUGI_TAHUN_BERJALAN_(ANN)","ROA","ROE","LRP","LLR",
      "OPERATION_COST","TOTAL_INCOME","CIR","INT_REV","INT_COST","INT_REV_ANN","INT_COST_ANN","RG_3_1","RG_3_2","RG_3_3","RG_3"
    ).filterNot(p => p.equals("ID_LAPORAN1") || p.equals("NAMA_BANK") || p.equals("TAHUN") )

    val db = mongoClient("bank_dataset")
    val repricingCollection = db("datasetrepricing_gap_1")

    val labelPosition = if(parameterBank.targetClassPosition == -1) featuresName.length - 1 else parameter.targetClassPosition

    println(repricingCollection.find().toList.size)

    val tempDataRG  = repricingCollection.find().toList.map( p => {
      featuresName.zipWithIndex.map( x =>( x._1 -> p.getAs[Double](x._1).getOrElse(p.getAs[Int](x._1).get.toDouble))).toMap
    })

    val tempDataset = tempDataRG.map { p =>
      featuresName.zipWithIndex.map { x =>
        if(x._2 == labelPosition) {
          BinaryValue(List(p.get(x._1).get)).asInstanceOf[Denomination[_]]
        }
        else {
          ContValue(p.get(x._1).get).asInstanceOf[Denomination[_]]
        }
      } toArray
    }

//    tempDataset.foreach { p=>
//      p.foreach( x => print(if(x.isInstanceOf[ContValue]) "*" + x.asInstanceOf[ContValue].get else "&"+x.asInstanceOf[BinaryValue].get
//        + "; "))
//      println("-")
//    }


    val alldataset = StandardNormalization.normalize(
      tempDataset, tempDataset
      , labelPosition, true)


        alldataset.foreach { p=>
          p.foreach( x => print(if(x.isInstanceOf[ContValue]) "; " + x.asInstanceOf[ContValue].get else "; "+x.asInstanceOf[BinaryValue].get))
          println("-")
        }

    assert(alldataset.size ==10424)
    assert(alldataset(0).size == featuresName.size)

  }
}
