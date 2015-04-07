/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

import scala.annotation.tailrec

/**
 * @author Eko Khannedy
 * @since 2/28/15
 */
object CrossValidation extends Validation {

  /**
   * Cross dataset
   * @param dataset dataset
   * @param fault split size
   */
  def cross(dataset: List[Array[Double]], fault: Int): List[(List[Array[Double]], List[Array[Double]])] = {
    val splitSize = dataset.length / fault

    @tailrec
    def split(iteration: Int, result: List[List[Array[Double]]], data: List[Array[Double]]): List[List[Array[Double]]] = {
      if (iteration > fault) {
        result
      } else {
        val (a, b) = data.splitAt(splitSize)
        split(iteration + 1, a :: result, b)
      }
    }

    val dataCross = split(1, List(), dataset)
    val finalDataCross = dataCross.map { data =>
      data -> dataCross.filterNot(_ == data).flatMap(a => a)
    }

    finalDataCross
  }

}
