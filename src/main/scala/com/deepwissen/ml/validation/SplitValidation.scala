/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.validation

import com.deepwissen.ml.utils.Denomination

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

/**
 * Split Validation
 * @author Hendri Karisma
 * @since 8/28/15
 */
object SplitForBankSequence {
  /**
   * Split dataset with given percentage (ex 70-30)
   * @param dataset dataset
   * @param size percentage (ex: 70,30)
   * @return pair of (training_dataset, classification_dataset)
   */
  def split(dataset: List[Array[Denomination[_]]], size: (Int, Int), year : Double): (List[Array[Denomination[_]]], List[Array[Denomination[_]]]) = {
    val (trainingLength, classificationLength) = size
    val splitAt = dataset.length * trainingLength / (trainingLength + classificationLength)
    dataset.splitAt(splitAt)
  }


  /**
   * Split dataset with given percentage (ex 70-30)
   * @param dataset dataset
   * @param year take data by time
   * @return pair of (training_dataset, classification_dataset)
   */
  def split(dataset: List[Map[String, Double]], fieldName : String ,year : Double): (List[Map[String, Double]], List[Map[String, Double]]) = {
    val trainingDataset = dataset.filter(p => (p.get(fieldName).get < year))
    val testongDataset = dataset.filter(p => (p.get(fieldName).get >= year))
//    val splitAt = dataset.length * trainingLength / (trainingLength + classificationLength)
    (trainingDataset,testongDataset)
  }


}