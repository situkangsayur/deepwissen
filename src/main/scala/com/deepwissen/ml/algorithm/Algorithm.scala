/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

/**
 * Base trait for Deep Learning Algorithm
 * @author Eko Khannedy
 * @since 2/25/15
 * @tparam DATASET dataset for training
 * @tparam DATA data for classification
 */
trait Algorithm[DATASET, DATA, PARAM <: TrainingParameter, MODEL] {

  /**
   * Run training with given dataset
   * @param dataset dataset
   * @param parameter parameter
   * @return model
   */
  def train(dataset: DATASET, parameter: PARAM): MODEL

}
