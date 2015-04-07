/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.statistics

import org.scalatest.FunSuite

/**
 * @author Eko Khannedy
 * @since 2/27/15
 */
class StandardRegression$Test extends FunSuite {

  test("standard regression") {
    val prediction = StandardRegression.train(List(
      1.0 -> 1.0,
      2.0 -> 2.0,
      3.0 -> 3.0,
      4.0 -> 4.0,
      5.0 -> 5.0,
      6.0 -> 6.0,
      7.0 -> 7.0,
      8.0 -> 8.0
    ))

    println(prediction.predict(9.0))
    println(prediction.predict(10.0))
    println(prediction.predict(11.0))
    println(prediction.predict(12.0))
  }

  test("pengangguran"){
    val prediction = StandardRegression.train(List(
      1986.0 -> 1.82,
      1987.0 -> 1.82,
      1988.0 -> 2.04,
      1989.0 -> 2.04,
      1990.0 -> 1.91,
      1991.0 -> 1.99,
      1992.0 -> 2.14,
      1993.0 -> 2.20,
      1994.0 -> 3.64,
      1995.0 -> 3.85,
      1996.0 -> 4.28,
      1997.0 -> 4.18,
      1998.0 -> 5.05,
      1999.0 -> 6.03,
      2000.0 -> 5.81,
      2001.0 -> 8.01,
      2002.0 -> 9.13,
      2003.0 -> 9.94,
      2004.0 -> 10.25,
      2005.0 -> 11.90,
      2006.0 -> 10.93,
      2007.0 -> 10.01,
      2008.0 -> 9.39,
      2009.0 -> 8.96,
      2010.0 -> 8.32,
      2011.0 -> 7.70,
      2012.0 -> 7.24,
      2013.0 -> 7.39
    ))

    println(prediction.predict(2014.0))
    println(prediction.predict(2015.0))
    println(prediction.predict(2016.0))
    println(prediction.predict(2017.0))
    println(prediction.predict(2018.0))
    println(prediction.predict(2019.0))
    println(prediction.predict(2020.0))
  }

}
