/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * Perceptron model class
 * @author Eko Khannedy
 * @since 2/25/15
 * @param id perceptron id
 * @param index index position in layer
 * @param output output value
 * @param weight weight value
 * @param error error value
 */
case class Perceptron(var id: String,
                      var index: Int,
                      var output: Double = 0.0,
                      var weight: Double = 0.0,
                      var error: Double = 0.0)
