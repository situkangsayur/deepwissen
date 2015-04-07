/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * @author Eko Khannedy
 * @since 2/25/15
 */
case class Synapsys(from: String,
                    to: String,
                    var weight: Double,
                    var deltaWeight: Double = 0.0) {

  val isFromBias = from.startsWith("bias_")

}
