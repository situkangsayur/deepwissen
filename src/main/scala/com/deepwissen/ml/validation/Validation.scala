/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

import com.deepwissen.ml.algorithm._
import com.deepwissen.ml.function.{ActivationFunction, ThresholdFunction}
import com.deepwissen.ml.utils.{BinaryValue, Denomination}

/**
 * Base trait for algorithm validation
 * @author Eko Khannedy
 * @since 2/27/15
 */
trait Validation {

  /**
   * Run classification with given dataset
   * @param network network model
   * @param classification classification
   * @param dataset dataset
   * @return list of result
   */
  def classification(network: Network, classification: Classification[Array[Denomination[_]], Network], dataset: List[Array[Denomination[_]]], activationFunction: ActivationFunction): List[Denomination[_]] =
    dataset.map(data => classification(data, network, activationFunction))

  /**
   * Validate classification result
   * @param result classification results
   * @param dataset dataset
   * @param targetClass targetClass index
   * @return validate result
   */
  def validate(result: List[Denomination[_]], dataset: List[Array[Denomination[_]]], targetClass: Int): List[(Denomination[_], Denomination[_])] = {
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
  def accuration(validateResult: List[(Denomination[_], Denomination[_])])(implicit thresholdFunction: ThresholdFunction): Double = {
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

object Validation extends Validation {

}
