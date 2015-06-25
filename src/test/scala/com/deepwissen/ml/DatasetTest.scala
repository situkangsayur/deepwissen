/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml

import org.scalatest.FunSuite

/**
 * @author Eko Khannedy
 * @since 3/1/15
 */
class DatasetTest extends FunSuite {

  test("test weather dataset") {
    Dataset.weatherLines.foreach(item => println(item))
    Dataset.weather.foreach(item => println(item.mkString(" | ")))
    Dataset.weatherRealDataset.foreach(item => println(item.mkString(" | ")))
//    Dataset.weatherDataset.foreach(item => println(item.mkString(" | ")))
  }

  test("test gdp dataset") {
    Dataset.gdpLines.foreach(item => println(item))
    Dataset.gdpRealDataset.foreach(item => println(item.mkString(" | ")))
//    Dataset.gdpDataset.foreach(item => println(item.mkString(" | ")))
  }

}
