/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

/**
 * Split Validation
 * @author Eko Khannedy
 * @since 2/27/15
 */
object SplitValidation {

  /**
   * Split dataset with given percentage (ex 70-30)
   * @param dataset dataset
   * @param size percentage (ex: 70,30)
   * @return pair of (training_dataset, classification_dataset)
   */
  def split(dataset: List[Array[Double]], size: (Int, Int)): (List[Array[Double]], List[Array[Double]]) = {
    val (trainingLength, classificationLength) = size
    val splitAt = dataset.length * trainingLength / (trainingLength + classificationLength)
    dataset.splitAt(splitAt)
  }

}
