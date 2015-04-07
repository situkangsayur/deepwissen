/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.statistics

/**
 * @author Eko Khannedy
 * @since 2/27/15
 */
object StandardRegression extends Regression {

  /**
   * Simple regression prediction
   * @param theta1 theta 1
   * @param theta2 theta 2
   */
  private case class StandardRegressionPrediction(theta1: Double, theta2: Double) extends RegressionPrediction {

    /**
     * Predict value
     * @param value value
     * @return predicted value
     */
    override def predict(value: Double): Double =
      (theta2 * value) + theta1
  }

  /**
   * Train dataset with linear regression
   * @param dataset dataset
   * @return regression prediction
   */
  override def train(dataset: List[(Double, Double)]): RegressionPrediction = {
    val theta2 = ((dataset.size * dataset.map(x => x._1 * x._2).foldLeft(0.0D)(_ + _)) -
      (dataset.map(x => x._1).foldLeft(0.0D)(_ + _) * dataset.map(x => x._2).foldLeft(0.0D)(_ + _))) /
      (dataset.size * dataset.map(x => math.pow(x._1, 2)).foldLeft(0.0D)(_ + _) -
        math.pow(dataset.map(x => x._1).foldLeft(0.0D)(_ + _), 2))
    val theta1 = (dataset.map(x => x._2).foldLeft(0.0D)(_ + _) - (theta2 * dataset.map(x => x._1).foldLeft(0.0D)(_ + _))) / dataset.size

    StandardRegressionPrediction(theta1, theta2)
  }
}
