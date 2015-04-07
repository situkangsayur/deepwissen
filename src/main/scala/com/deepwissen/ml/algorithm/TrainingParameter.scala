/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * Base trait for training parameter
 * @author Eko Khannedy
 * @since 2/25/15
 */
trait TrainingParameter

/**
 * Parameter for backpropragation algorithm
 */
case class BackpropragationParameter(hiddenLayerSize: Int = 2, learningRate: Double = 0.5,
                                     momentum: Double = 0.75, epsilon: Double = 0.000001,
                                     iteration: Int = Int.MaxValue) extends TrainingParameter
