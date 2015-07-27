/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.normalization

import com.deepwissen.ml.algorithm.TrainingParameter
import com.deepwissen.ml.utils.{BinaryValue, ContValue, Denomination}

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
  override def  normalize(dataset: List[Array[Denomination[_]]], targetClass: Int): List[Array[Denomination[_]]] = {
    val minMax: Map[Int, (Double, Double)] = dataset.head.indices.filter(p => p != targetClass).map { i =>
      val min = dataset.foldLeft(Double.MaxValue) { (value, current) =>
        if (value > current(i).asInstanceOf[ContValue].get) current(i).asInstanceOf[ContValue].get else value
      }
      val max = dataset.foldLeft(Double.MinValue) { (value, current) =>
        if (value > current(i).asInstanceOf[ContValue].get) value else current(i).asInstanceOf[ContValue].get
      }

      i ->(min, max)
    }.toMap

    dataset.map { array =>
      dataset.head.indices.map { i =>
        val temp: Denomination[_] = if(i != targetClass) {
          val min = if(minMax(i)._1 == minMax(i)._2) 0 else minMax(i)._1
          val max = if(minMax(i)._1 == minMax(i)._2) 1 else minMax(i)._2

          normalize(array(i).asInstanceOf[ContValue].get, min, max, targetClass)
        } else array(i)

        temp
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
  override def denormalize(normalValue: Double, index: Int, dataset: List[Array[Denomination[_]]], targetClass: Int): Double = {
    val min = dataset.foldLeft(Double.MaxValue) { (value, current) =>
      if (value > current(index).asInstanceOf[ContValue].get) current(index).asInstanceOf[ContValue].get else value
    }
    val max = dataset.foldLeft(Double.MinValue) { (value, current) =>
      if (value > current(index).asInstanceOf[ContValue].get) value else current(index).asInstanceOf[ContValue].get
    }

    denormalize(normalValue, min, max, targetClass)
  }
}