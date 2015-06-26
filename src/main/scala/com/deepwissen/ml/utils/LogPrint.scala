package com.deepwissen.ml.utils

import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 6/26/15.
 */
object LogPrint {

  var logger  = LoggerFactory.getLogger("Main Objects")

  def printLogDebug(str : String): Unit = {
//    println("#Debug# "+str)
  }

  def printLogError(str : String): Unit = {
    println("#Error# "+str)
  }

  def printLogInfo(str : String): Unit = {
    println("#Info# "+str)
  }
}
