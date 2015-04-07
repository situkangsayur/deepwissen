/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.function.{SigmoidFunction, ActivationFunction}

/**
 * @author Eko Khannedy
 * @since 2/25/15
 */
case class Perceptron(id: String,
                      index: Int,
                      var output: Double = 0.0,
                      var weight: Double = 0.0,
                      var error: Double = 0.0,
                      var activationFunction: ActivationFunction = SigmoidFunction)
