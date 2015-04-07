/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.priorknowledge

/**
 * Standard prior knowledge for List of Array String
 * @author Eko Khannedy
 * @since 3/1/15
 */
case class StandardPriorKnowledge(dataset: List[Array[String]]) extends PriorKnowledge[String, Int] {

  private val lookupData: Map[Int, Map[String, Int]] = (0 until dataset.head.length).map { i =>
    val columnData = dataset.map(array => array(i))
    val columnDataDistict = columnData.distinct
    val columnDataPair = columnDataDistict.zipWithIndex.toMap

    i -> columnDataPair
  }.toMap

  /**
   * Lookup data
   * @param value data
   * @param index primary key
   * @return option double
   */
  override def lookup(value: String, index: Int): Option[Double] =
    lookupData(index).get(value).map(_.toDouble)
}
