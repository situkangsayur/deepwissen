/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

import com.deepwissen.ml.algorithm._
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
  def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): Double

}


/**
 * Normal Backpro validation
 * @author Hendri Karisma
 * @since 7/25/15
 */
case class BackProValidation() extends Validation[Network, List[Array[Denomination[_]]]] {

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
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): Double = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).map(x => {
      val  temp = x.map { case (score, target) =>
        thresholdFunction.compare(score, target)
      } filter(p => p == false)
      if(temp.size == 0) true else false
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b)
    val accuration = 100.0 / totalData * totalCorrect
    accuration
  }
}

/**
 * Normal Autoencoder validation
 * @author Hendri Karisma
 * @since 7/25/15
 */
case class AutoencoderValidation() extends Validation[Network, List[Array[Denomination[_]]]] {

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
      value -> BinaryValue(dataset(index).zipWithIndex.filter(p => p._2 != targetClass).map(data => data._1.asInstanceOf[ContValue].get).toList)
    }
  }

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): Double = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).flatMap(x => {
      x.map { case (score, target) =>
        thresholdFunction.compare(score, target)
      }
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b)
    val accuration = 100.0 / totalData * totalCorrect
    accuration
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
  override def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): Double = {
    val compareResult = validateResult.map(x => x._1.asInstanceOf[BinaryValue].get.zip(x._2.asInstanceOf[BinaryValue].get)).flatMap(x => {
      x.map { case (score, target) =>
        thresholdFunction.compare(score, target)
      }
    })
    val totalData = compareResult.length
    val totalCorrect = compareResult.count(b => b)
    val accuration = (100.0 / totalData) * totalCorrect
    accuration
  }
}