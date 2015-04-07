/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.priorknowledge

/**
 * Base trait for prior knowledge
 * @author Eko Khannedy
 * @since 3/1/15
 * @tparam V value data
 */
trait PriorKnowledge[V, PK] {

  /**
   * Lookup data 
   * @param value data
   * @param index primary key
   * @return option double
   */
  def lookup(value: V, index: PK): Option[Double]

}
