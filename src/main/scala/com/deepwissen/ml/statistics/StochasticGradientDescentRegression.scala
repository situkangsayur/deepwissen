/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.statistics

import com.deepwissen.ml.statistics.StochasticGradientDescentRegression.Parameter

/**
 * @author Eko Khannedy
 * @since 2/27/15
 */
case class StochasticGradientDescentRegression(parameter: Parameter) extends Regression {

  /**
   * StochasticGradientDescentRegressionPrediction
   * @param theta1 theta 1
   * @param theta2 theta 2
   */
  private case class StochasticGradientDescentRegressionPrediction(theta1: Double, theta2: Double) extends RegressionPrediction {

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

    // param
    var theta1: Double = parameter.theta1
    var theta2: Double = parameter.theta2
    val epsilon: Double = parameter.epsilon
    val maxEpoch: Int = parameter.maxEpoch
    val alpha: Double = parameter.alpha

    def predict(value: Double): Double = (theta2 * value) + theta1

    var cost: Double = 1
    var iter = 0
    var error = 0.0D

    do {
      error = (1.0D / (2 * dataset.size)) * dataset.map(x => math.pow(predict(x._1) - x._2, 2)).foldLeft(0.0D)(_ + _)
      cost = (1.0D / dataset.size) * dataset.map(x => predict(x._1) - x._2).foldLeft(0.0D)(_ + _)
      theta1 = theta1 - (alpha * (1.0D / dataset.size) * dataset.map(x => predict(x._1) - x._2).foldLeft(0.0D)(_ + _))
      theta2 = theta2 - (alpha * cost)
      iter += 1
    } while ((error > epsilon) && (iter <= maxEpoch))

    StochasticGradientDescentRegressionPrediction(theta1, theta2)
  }
}

object StochasticGradientDescentRegression {

  /**
   * StochasticGradientDescentRegression Parameter
   * @param theta1 theta1
   * @param theta2 theta2
   * @param epsilon epsilon
   * @param maxEpoch maxEpoch
   * @param alpha alpha
   */
  case class Parameter(theta1: Double = 0.5D,
                       theta2: Double = 1.0D,
                       epsilon: Double = 0.00000001D,
                       maxEpoch: Int = 200000,
                       alpha: Double = 0.1D)
}