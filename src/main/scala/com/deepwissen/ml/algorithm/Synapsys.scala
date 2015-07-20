/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * @author Eko Khannedy
 * @since 2/25/15
 */
case class Synapsys(var from: Perceptron,
                    var to: Perceptron,
                    var weight: Double,
                    var deltaWeight: Double = 0.0) {

  /**
   * Is synspsys from bias perceptron
   */
  val isFromBias = from.id.startsWith("bias_")

}
