/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

import com.deepwissen.ml.algorithm._
import com.deepwissen.ml.algorithm.networks.{DeepNetwork, AutoencoderNetwork, Network, MarkovChain}
import com.deepwissen.ml.function.{ActivationFunction, ThresholdFunction}
import com.deepwissen.ml.utils.{ContValue, BinaryValue, Denomination}

/**
 * Base trait for algorithm validation
 * @author Eko Khannedy
 * @since 2/27/15
 */
trait Validation[T, U] {

  /**
   * Run classification with given dataset
   * @param network network model
   * @param classification classification
   * @param dataset dataset
   * @return list of result
   */
  def classification(network: T, classification: Classification[Array[Denomination[_]], T], dataset: List[Array[Denomination[_]]], activationFunction: ActivationFunction): List[Denomination[_]]

  /**
   * Validate classification result
   * @param result classification results
   * @param dataset dataset
   * @param targetClass targetClass index
   * @return validate result
   */
  def validate(result: List[Denomination[_]], dataset: U, targetClass: Int): List[(Denomination[_], Denomination[_])]

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): (Double, Double, Double)

}


/**
 * Normal Backpro validation
 * @author Hendri Karisma
 * @since 7/25/15
 */
case class BackProValidation(tL : Double = 0.6, tE : Double = 0.6, k : Double = 1.0) extends Validation[Network, List[Array[Denomination[_]]]] {

  /**
   * Run classification with given dataset
   * @param network network model
   * @param classification classification
   * @param dataset dataset
   * @return list of result
   */
  override def classification(network: Network, classification: Classification[Array[Denomination[_]], Network], dataset: List[Array[Denomination[_]]], activationFunction: ActivationFunction): List[Denomination[_]] =
    dataset.map(data => {
      classification(data, network, activationFunction)
    })

  override def validate(result: List[Denomination[_]], dataset: List[Array[Denomination[_]]], targetClass: Int): List[(Denomination[_], Denomination[_])] = {
    result.zipWithIndex.map { case (value, index) =>
      value -> (dataset(index)(targetClass))
    }
  }

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): (Double, Double, Double) = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).map(x => {
      val  temp = x.map { case (score, target) =>
        val temp = thresholdFunction.compare(score, target)
        (temp._1, temp._2, score, target)
      }

      //alpha parameter
      val tempLoss = temp.foldLeft(0.0D)((temp, sc) => temp + math.pow(sc._2,2))/temp.size
      val iFunc = if(tempLoss >= tL) 1 else 0
      val alpha = iFunc * (1 - math.exp((-k) * (math.pow(tempLoss - tL, 2)/math.pow(tL,2))))

      //recall parameter
      val yCount = temp.foldLeft(0.0D)((temp, sc) => temp + sc._4) / temp.size
      val tempTethaY = 1.0D / (1.0D + math.exp(-1.0 * yCount))
      val tethaY = if (tempTethaY >= tE) 1 else 0

      //precision parameter
      val yTiltCount = temp.foldLeft(0.0D)((temp, sc) => temp + sc._3) / temp.size
      val tempTethaYTilt = 1.0D / (1.0D + math.exp(-1.0 * yCount))
      val tethaYTilt = if (tempTethaY >= tE) 1 else 0


      val tempResult = if(temp.filter(p => p._1 == false).size == 0) true else false
      (tempResult, alpha, tethaY,tethaYTilt)
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b._1)
    val recall = compareResult.foldLeft(0.0D)((temp, data) => temp + data._2 * data._3) /
      compareResult.foldLeft(0.0D)((temp, data) => temp + data._3)

    val precision = compareResult.foldLeft(0.0D)((temp, data) => temp + data._2 * data._4) /
      compareResult.foldLeft(0.0D)((temp, data) => temp + data._4)

    val accuration = 100.0 / totalData * totalCorrect
    (accuration, recall, precision)
  }
}

/**
 * Normal Autoencoder validation
 * @author Hendri Karisma
 * @since 7/25/15
 */
case class AutoencoderValidation() extends Validation[AutoencoderNetwork, List[Array[Denomination[_]]]] {

  /**
   * Run classification with given dataset
   * @param network network model
   * @param classification classification
   * @param dataset dataset
   * @return list of result
   */
  override def classification(network: AutoencoderNetwork, classification: Classification[Array[Denomination[_]], AutoencoderNetwork], dataset: List[Array[Denomination[_]]], activationFunction: ActivationFunction): List[Denomination[_]] =
    dataset.map(data => {
      classification(data, network, activationFunction)
    })

  override def validate(result: List[Denomination[_]], dataset: List[Array[Denomination[_]]], targetClass: Int): List[(Denomination[_], Denomination[_])] = {
    result.zipWithIndex.map { case (value, index) =>
      value -> BinaryValue(dataset(index).zipWithIndex.filter(p => p._2 != targetClass).map(data => data._1.asInstanceOf[ContValue].get).toList)
    }
  }

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): (Double, Double, Double) = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).flatMap(x => {
      x.map { case (score, target) =>
        thresholdFunction.compare(score, target)
      }
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b._1)
    val accuration = 100.0 / totalData * totalCorrect
    (accuration, 0.0, 0.0)
  }
}

/**
 * RBM Algorithm validation
 * @author Hendri Karisma
 * @since 7/25/15
 */
case class MarkovChainValidation() extends Validation[MarkovChain, List[Array[Denomination[_]]]] {
  /**
   * Run classification with given dataset
   * @param network network model
   * @param classification classification
   * @param dataset dataset
   * @return list of result
   */
  override def classification(network: MarkovChain, classification: Classification[Array[Denomination[_]], MarkovChain], dataset: List[Array[Denomination[_]]], activationFunction: ActivationFunction): List[Denomination[_]] =
    dataset.map(data => {
      classification(data, network, activationFunction)
    })

  override def validate(result: List[Denomination[_]], dataset: List[Array[Denomination[_]]], targetClass: Int): List[(Denomination[_], Denomination[_])] = {
    result.zipWithIndex.map { case (value, index) =>
      value -> BinaryValue(dataset(index).zipWithIndex.filter(p => p._2 != targetClass).map(data => data._1.asInstanceOf[ContValue].get).toList)
    }
  }

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): (Double, Double, Double) = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).flatMap(x => {
      x.map { case (score, target) =>
        thresholdFunction.compare(score, target)
      }
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b._1)
    val accuration = (100.0 / totalData) * totalCorrect
    (accuration, 0.0, 0.0)
  }
}


/**
 * Normal Deep Network validation
 * @author Hendri Karisma
 * @since 7/25/15
 */
case class DeepNetworkValidation(tL : Double = 0.6, tE : Double = 0.6, k : Double = 1.0) extends Validation[DeepNetwork, List[Array[Denomination[_]]]] {

  /**
   * Run classification with given dataset
   * @param network network model
   * @param classification classification
   * @param dataset dataset
   * @return list of result
   */
  override def classification(network: DeepNetwork, classification: Classification[Array[Denomination[_]], DeepNetwork], dataset: List[Array[Denomination[_]]], activationFunction: ActivationFunction): List[Denomination[_]] =
    dataset.map(data => {
      classification(data, network, activationFunction)
    })

  override def validate(result: List[Denomination[_]], dataset: List[Array[Denomination[_]]], targetClass: Int): List[(Denomination[_], Denomination[_])] = {
    result.zipWithIndex.map { case (value, index) =>
      value -> (dataset(index)(targetClass))
    }
  }

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): (Double, Double, Double) = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).map(x => {
      val  temp = x.map { case (score, target) =>
        val temp = thresholdFunction.compare(score, target)
        (temp._1, temp._2, score, target)
      }

      //alpha parameter
      val tempLoss = temp.foldLeft(0.0D)((temp, sc) => temp + math.pow(sc._2,2))/temp.size
      val iFunc = if(tempLoss >= tL) 1 else 0
      val alpha = iFunc * (1 - math.exp((-k) * (math.pow(tempLoss - tL, 2)/math.pow(tL,2))))

      //recall parameter
      val yCount = temp.foldLeft(0.0D)((temp, sc) => temp + sc._4) / temp.size
      val tempTethaY = 1.0D / (1.0D + math.exp(-1.0 * yCount))
      val tethaY = if (tempTethaY >= tE) 1 else 0

      //precision parameter
      val yTiltCount = temp.foldLeft(0.0D)((temp, sc) => temp + sc._3) / temp.size
      val tempTethaYTilt = 1.0D / (1.0D + math.exp(-1.0 * yCount))
      val tethaYTilt = if (tempTethaY >= tE) 1 else 0


      val tempResult = if(temp.filter(p => p._1 == false).size == 0) true else false
      (tempResult, alpha, tethaY,tethaYTilt)
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b._1)
    val recall = compareResult.foldLeft(0.0D)((temp, data) => temp + data._2 * data._3) /
      compareResult.foldLeft(0.0D)((temp, data) => temp + data._3)

    val precision = compareResult.foldLeft(0.0D)((temp, data) => temp + data._2 * data._4) /
      compareResult.foldLeft(0.0D)((temp, data) => temp + data._4)

    val accuration = 100.0 / totalData * totalCorrect
    (accuration, recall, precision)
  }
}