/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.normalization

import com.deepwissen.ml.algorithm.TrainingParameter
import com.deepwissen.ml.utils.{TargetValue, FieldValue, Denomination}

/**
 * Normalization for List[Array[Double]]
 * @author Eko Khannedy
 * @since 3/1/15
 */
object StandardNormalization extends Normalization[List[Array[Denomination[_]]]] {

  /**
   * Normalize dataset
   * @param dataset dataset
   * @return normal dataset
   */
  override def normalize(dataset: List[Array[Denomination[_]]], targetClass: Int): List[Array[Denomination[_]]] = {
    val minMax: Map[Int, (Double, Double)] = dataset.head.indices.filter(p => p != targetClass).map { i =>
      val min = dataset.foldLeft(Double.MaxValue) { (value, current) =>
        if (value > current(i).asInstanceOf[FieldValue].get) current(i).asInstanceOf[FieldValue].get else value
      }
      val max = dataset.foldLeft(Double.MinValue) { (value, current) =>
        if (value > current(i).asInstanceOf[FieldValue].get) value else current(i).asInstanceOf[FieldValue].get
      }

      i ->(min, max)
    }.toMap

    dataset.map { array =>
      dataset.head.indices.map { i =>
        if(i != targetClass) {
          val min = minMax(i)._1
          val max = minMax(i)._2
          normalize(array(i).asInstanceOf[FieldValue].get, min, max, targetClass)
        }else array(i).asInstanceOf[TargetValue]
      }.toArray.asInstanceOf[Array[Denomination[_]]]
    }
  }

  /**
   * Denormalize normal value
   * @param normalValue normal value
   * @param index index of value
   * @param dataset dataset
   * @return double
   */
  override def denormalize(normalValue: Double, index: Int, dataset: List[Array[Denomination[_]]], targetClass: Int): Double = {
    val min = dataset.foldLeft(Double.MaxValue) { (value, current) =>
      if (value > current(index).asInstanceOf[FieldValue].get) current(index).asInstanceOf[FieldValue].get else value
    }
    val max = dataset.foldLeft(Double.MinValue) { (value, current) =>
      if (value > current(index).asInstanceOf[FieldValue].get) value else current(index).asInstanceOf[FieldValue].get
    }

    denormalize(normalValue, min, max, targetClass)
  }
}