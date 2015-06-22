/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

import com.deepwissen.ml.algorithm._
import com.deepwissen.ml.function.{ActivationFunction, ThresholdFunction}

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
  def classification(network: Network, classification: Classification[Array[Any], Network], dataset: List[Array[Any]], activationFunction: ActivationFunction): List[List[Double]] =
    dataset.map(data => classification(data, network, activationFunction))

  /**
   * Validate classification result
   * @param result classification results
   * @param dataset dataset
   * @param targetClass targetClass index
   * @return validate result
   */
  def validate(result: List[List[Double]], dataset: List[Array[Any]], targetClass: Int): List[(List[Double], List[Double])] = {
    result.zipWithIndex.map { case (value, index) =>
      value -> (dataset(index)(targetClass)).asInstanceOf[List[Double]]
    }
  }

  /**
   * Calculate accuration of validation result
   * @param validateResult validation result
   * @param thresholdFunction threshold function
   * @return accuration
   */
  def accuration(validateResult: List[(List[Double], List[Double])])(implicit thresholdFunction: ThresholdFunction): Double = {
    val compareResult = validateResult.map(x => x._1.zip(x._2)).map(x => {
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
