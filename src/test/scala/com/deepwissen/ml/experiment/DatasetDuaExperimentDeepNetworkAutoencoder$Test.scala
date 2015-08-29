package com.deepwissen.ml.experiment

import java.io.{File, FileOutputStream}

import com.deepwissen.ml.algorithm._
import com.deepwissen.ml.function.{RangeThresholdFunction, EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{BinaryValue, ContValue, Denomination}
import com.deepwissen.ml.validation.{SplitForBankSequence, DeepNetworkValidation, BackProValidation}
import com.mongodb.casbah.Imports._
import com.mongodb.casbah.MongoClient
import com.mongodb.casbah.commons.MongoDBObject
import org.scalatest.FunSuite

/**
 * Created by hendri_k on 8/28/15.
 */
class DatasetDuaExperimentDeepNetworkAutoencoder$Test extends FunSuite{

  val mongoClient =  MongoClient()

  test("test for experiments dataset satu with Deep Network Autoencoder"){


    val tempFeaturesName = List("ID_BANK","ID_LAPORAN1","NAMA_BANK","TAHUN","BULAN","Illiquid_Assets","Illiquid_Liabilities",
      "LTR","Giro","Tabungan","Deposito","DPK","CASA","CORE_DEPOSITS","Kredit","FINANCING_GAP","TOTAL_ASET","ATMR","RWA",
      "CAR","TotalEkuitas","EQTA","LABA_RUGI_TAHUN_BERJALAN","LABA_RUGI_TAHUN_BERJALAN_(ANN)","ROA","ROE","LRP","LLR",
      "OPERATION_COST","TOTAL_INCOME","CIR","INT_REV","INT_COST","INT_REV_ANN","INT_COST_ANN","RG_6_1","RG_6_2","RG_6_3",
      "RG_6_4","RG_6_5","RG_6_6","RG_6"
    ).filterNot(p => p.equals("ID_LAPORAN1") || p.equals("NAMA_BANK") || p.equals("TAHUN") )

    val db = mongoClient("bank_dataset")
    val repricingCollection = db("datasetrepricing_gap_2")

    println(repricingCollection.find("TAHUN" $gte 2007).toList.size)

    val tempDataRG  = repricingCollection.find("TAHUN" $gte 2007).map( p => {
      tempFeaturesName.zipWithIndex.map( x =>( x._1 -> p.getAs[Double](x._1).getOrElse(p.getAs[Int](x._1).get.toDouble))).toMap
    }).toList

    val datasetRG = SplitForBankSequence.split(dataset = tempDataRG, fieldName = "TAHUN", year = 2013)

    val featuresName = tempFeaturesName.filterNot(p => p.equals("TAHUN"))

    /**
     * Training Parameter
     */
    val parameterBank = DeepNetworkParameter(
      //    hiddenLayerSize = List(9,10,11,12,11,10,9),
      //    hiddenLayerSize = List(11,11, 11, 11, 11, 11),
      hiddenLayerSize = List(35,35,35,35) ,
      outputPerceptronSize = 1,
      targetClassPosition = -1,
      iteration = 500,
      epsilon = 0.00000001,
      momentum = 0.3,
      learningRate = 0.3,
      synapsysFactory = RandomSynapsysFactory(),
      activationFunction = SigmoidFunction,
      inputPerceptronSize = featuresName.size - 1,
      autoecoderParam = AutoencoderParameter(
        iteration = 20,
        epsilon = 0.00001,
        momentum = 0.50,
        learningRate = 0.50,
        synapsysFactory = RandomSynapsysFactory(),
        activationFunction = SigmoidFunction
      )
    )



    val labelPosition = if(parameterBank.targetClassPosition == -1) featuresName.length - 1 else parameterBank.targetClassPosition

    val tempDatasetTraining = datasetRG._1.map { p =>
      featuresName.zipWithIndex.map { x =>
        if(x._2 == labelPosition) {
          BinaryValue(List(p.get(x._1).get)).asInstanceOf[Denomination[_]]
        }
        else {
          ContValue(p.get(x._1).get).asInstanceOf[Denomination[_]]
        }
      } toArray
    }

    val tempDatasetTesting = datasetRG._2.map { p =>
      featuresName.zipWithIndex.map { x =>
        if(x._2 == labelPosition) {
          BinaryValue(List(p.get(x._1).get)).asInstanceOf[Denomination[_]]
        }
        else {
          ContValue(p.get(x._1).get).asInstanceOf[Denomination[_]]
        }
      } toArray
    }

    val datasetTraining = StandardNormalization.normalize(
      tempDatasetTraining
      , labelPosition, true)

    val datasetTesting = StandardNormalization.normalize(
      tempDatasetTesting
      , labelPosition, true)



    //    alldataset.foreach { p=>
//      p.foreach( x => print(if(x.isInstanceOf[ContValue]) "; " + x.asInstanceOf[ContValue].get else "; "+x.asInstanceOf[BinaryValue].get))
//      println("-")
//    }

//    assert(alldataset.size ==6992)
//    assert(alldataset(0).size == featuresName.size)
    assert(datasetTraining.size ==3744)
    assert(datasetTraining(0).size == featuresName.size)
    assert(datasetTesting.size ==624)
    assert(datasetTesting(0).size == featuresName.size)

    //test algoritma

    try {

      //      logger.info(finalDataSetBreastCancer.toString())

      val network = DeepNetworkAlgorithm.train(datasetTraining, parameterBank)

      val validator = DeepNetworkValidation(tE = 0.05,tL = 0.05, k = 2.3)

      val result = validator.classification(network, DeepNetworkClassification, datasetTesting, SigmoidFunction)
      //            logger.info("result finding : "+ result.toString())

      val validateResult = validator.validate(result, datasetTesting, labelPosition)


      val accuration = validator.accuration(validateResult) {
        EitherThresholdFunction(0.5, 0.0, 1.0)
      }

      val accurationRange = validator.accuration(validateResult) {
        RangeThresholdFunction(0.01)
      }



      val threshold = RangeThresholdFunction(0.01)


      var trueCounter = 0
      var allData = 0

      // classification
      datasetTesting.foreach { data =>
        val realScore = DeepNetworkClassification(data, network, SigmoidFunction)
        realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
          val originalClass = data(labelPosition).asInstanceOf[BinaryValue].get(0)
          val result = p._1
          val compare = threshold.compare(p._1, originalClass)
          println(s"real $p == score $compare == targetClass ${originalClass}")
          trueCounter = if(compare._1) trueCounter + 1 else trueCounter
          allData += 1
        })
        println("------------------------------------------------------------")
      }

      val percent = trueCounter * (100.0 / allData)

      println("result Either Threshold Function : " + accuration._1 +" :> recall : " + accuration._2 + " :> precision : " + accuration._3)
      println("result RangeThresholdFunction : " + accurationRange._1 +" :> recall : " + accurationRange._2 + " :> precision : " + accurationRange._3)

      println("result comparation : " + trueCounter + " :> in percent : " + percent)

      assert(percent >= 80)
      assert(accurationRange._1 >= 80)

      // save model
      NetworkSerialization.save(network, new FileOutputStream(
        new File("target" + File.separator + "bank_rg_data_2_dpa.json")))
    } catch {
      case npe: NullPointerException => npe.printStackTrace()
      case e: Exception => e.printStackTrace()
    }

  }
}
