/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.statistics

/**
 * @author Eko Khannedy
 * @since 2/27/15
 */
trait Regression {

  /**
   * Train dataset with linear regression
   * @param dataset dataset
   * @return regression prediction
   */
  def train(dataset: List[(Double, Double)]): RegressionPrediction

}

/**
 * Regression Prediction, this is result from Regression
 * @author Eko Khannedy
 * @since 2/27/15
 */
trait RegressionPrediction {

  /**
   * Predict value
   * @param value value
   * @return predicted value
   */
  def predict(value: Double): Double

}
