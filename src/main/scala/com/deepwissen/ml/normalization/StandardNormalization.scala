/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.normalization

/**
 * Normalization for List[Array[Double]]
 * @author Eko Khannedy
 * @since 3/1/15
 */
object StandardNormalization extends Normalization[List[Array[Double]]] {

  /**
   * Normalize dataset
   * @param dataset dataset
   * @return normal dataset
   */
  override def normalize(dataset: List[Array[Double]]): List[Array[Double]] = {
    val minMax: Map[Int, (Double, Double)] = (0 until dataset(0).length).map { i =>
      val min = dataset.foldLeft(Double.MaxValue) { (value, current) =>
        if (value > current(i)) current(i) else value
      }
      val max = dataset.foldLeft(Double.MinValue) { (value, current) =>
        if (value > current(i)) value else current(i)
      }

      i ->(min, max)
    }.toMap

    dataset.map { array =>
      (0 until dataset.head.length).map { i =>
        val min = minMax(i)._1
        val max = minMax(i)._2
        (array(i) - min) / (max - min)
      }.toArray
    }
  }

  /**
   * Denormalize normal value
   * @param normalValue normal value
   * @param index index of value
   * @param dataset dataset
   * @return double
   */
  override def denormalize(normalValue: Double, index: Int, dataset: List[Array[Double]]): Double = {
    val min = dataset.foldLeft(Double.MaxValue) { (value, current) =>
      if (value > current(index)) current(index) else value
    }
    val max = dataset.foldLeft(Double.MinValue) { (value, current) =>
      if (value > current(index)) value else current(index)
    }

    denormalize(normalValue, min, max)
  }
}