package com.deepwissen.ml.algorithm

import java.io.{FileInputStream, File, FileOutputStream}

import com.deepwissen.ml.algorithm.networks.Network
import com.deepwissen.ml.function.{EitherThresholdFunction, SigmoidFunction}
import com.deepwissen.ml.normalization.StandardNormalization
import com.deepwissen.ml.serialization.NetworkSerialization
import com.deepwissen.ml.utils.{Denomination, BinaryValue, ContValue}
import com.deepwissen.ml.validation.{DeepNetworkValidation, BackProValidation}
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

/**
 * Created by hendri_k on 7/25/15.
 */
class DeepNetworkAlgorithm$Test extends FunSuite{

  /**
   * breast cancer dataset priorknowledge
   */

  val age = Map(
    "10-19" -> ContValue(0.0),
    "20-29" -> ContValue(1.0),
    "30-39" -> ContValue(2.0),
    "40-49" -> ContValue(3.0),
    "50-59" -> ContValue(4.0),
    "60-69" -> ContValue(5.0),
    "70-79" -> ContValue(6.0),
    "80-89" -> ContValue(7.0),
    "90-99" -> ContValue(8.0),
    "?" -> ContValue(9.0)
  )

  val menopause = Map(
    "lt40" -> ContValue(0.0),
    "ge40" -> ContValue(1.0),
    "premeno" -> ContValue(2.0),
    "?" -> ContValue(3.0)
  )

  val tumorSize = Map(
    "0-4" -> ContValue(0.0),
    "5-9" -> ContValue(1.0),
    "10-14" -> ContValue(2.0),
    "15-19" -> ContValue(3.0),
    "20-24" -> ContValue(4.0),
    "25-29" -> ContValue(5.0),
    "30-34" -> ContValue(6.0),
    "35-39" -> ContValue(7.0),
    "40-44" -> ContValue(8.0),
    "45-49" -> ContValue(9.0),
    "50-54" -> ContValue(10.0),
    "55-59" -> ContValue(11.0),
    "?" -> ContValue(12.0)
  )

  val invNodes = Map(
    "0-2" -> ContValue(0.0),
    "3-5" -> ContValue(1.0),
    "6-8" -> ContValue(2.0),
    "9-11" -> ContValue(3.0),
    "12-14" -> ContValue(4.0),
    "15-17" -> ContValue(5.0),
    "18-20" -> ContValue(6.0),
    "21-23" -> ContValue(7.0),
    "24-26" -> ContValue(8.0),
    "27-29" -> ContValue(9.0),
    "30-32" -> ContValue(10.0),
    "33-35" -> ContValue(11.0),
    "36-39" -> ContValue(12.0),
    "?" -> ContValue(13.0)
  )

  val nodeCaps = Map(
    "yes" -> ContValue(0.0),
    "no" -> ContValue(1.0),
    "?" -> ContValue(2.0)
  )

  val degMalig = Map(
    "1" -> ContValue(0.0),
    "2" -> ContValue(1.0),
    "3" -> ContValue(2.0),
    "?" -> ContValue(3.0)
  )

  val breast = Map(
    "left" -> ContValue(0.0),
    "right" -> ContValue(1.0),
    "?" -> ContValue(2.0)
  )

  val breastQuad = Map(
    "left_up" -> ContValue(0.0),
    "left_low" -> ContValue(1.0),
    "right_up" -> ContValue(2.0),
    "right_low" -> ContValue(3.0),
    "central" -> ContValue(4.0),
    "?" -> ContValue(5.0)
  )

  val irradiat = Map(
    "yes" -> ContValue(0.0),
    "no" -> ContValue(1.0),
    "?" -> ContValue(2.0)
  )

  val classTarget = Map(
    "no-recurrence-events" -> BinaryValue(List(0.0)),
    "recurrence-events" -> BinaryValue(List(1.0))
  )


  val priorKnowledgeBreastCancer : List[Map[String, Denomination[_]]] =
    List(age, menopause,tumorSize,invNodes, nodeCaps, degMalig, breast, breastQuad, irradiat, classTarget)

  /**
   * play tennis dataset priorknowledge
   */
  val outlook = Map(
    "sunny" -> ContValue(0.0),
    "overcast" -> ContValue(1.0),
    "rainy" -> ContValue(2.0)
  )

  val temperature = Map(
    "hot" -> ContValue(0.0),
    "mild" -> ContValue(1.0),
    "cool" -> ContValue(2.0)
  )

  val humidity = Map(
    "high" -> ContValue(0.0),
    "normal" -> ContValue(1.0)
  )

  val windy = Map(
    "TRUE" -> ContValue(0.0),
    "FALSE" -> ContValue(1.0)
  )

  val play = Map(
    "no" -> BinaryValue(List(0.0,0.0)),
    "yes" -> BinaryValue(List(0.0,1.0))
  )

  val priorKnowledge: List[Map[String, Denomination[_]]] = List(outlook, temperature, humidity, windy, play)


  /**
   * dataset breast cancer
   */
  val stringsBreastCancer =
  """
    |40-49,premeno,15-19,0-2,yes,3,right,left_up,no,recurrence-events
    |50-59,ge40,15-19,0-2,no,1,right,central,no,no-recurrence-events
    |50-59,ge40,35-39,0-2,no,2,left,left_low,no,recurrence-events
    |40-49,premeno,35-39,0-2,yes,3,right,left_low,yes,no-recurrence-events
    |40-49,premeno,30-34,3-5,yes,2,left,right_up,no,recurrence-events
    |50-59,premeno,25-29,3-5,no,2,right,left_up,yes,no-recurrence-events
    |50-59,ge40,40-44,0-2,no,3,left,left_up,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,2,left,left_up,no,no-recurrence-events
    |40-49,premeno,0-4,0-2,no,2,right,right_low,no,no-recurrence-events
    |40-49,ge40,40-44,15-17,yes,2,right,left_up,yes,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,2,left,left_low,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,2,right,left_up,no,no-recurrence-events
    |50-59,ge40,30-34,0-2,no,1,right,central,no,no-recurrence-events
    |50-59,ge40,25-29,0-2,no,2,right,left_up,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,2,left,left_low,yes,recurrence-events
    |30-39,premeno,20-24,0-2,no,3,left,central,no,no-recurrence-events
    |50-59,premeno,10-14,3-5,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,2,right,left_up,no,no-recurrence-events
    |50-59,premeno,40-44,0-2,no,2,left,left_up,no,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,3,left,left_up,no,no-recurrence-events
    |50-59,lt40,20-24,0-2,?,1,left,left_low,no,recurrence-events
    |60-69,ge40,40-44,3-5,no,2,right,left_up,yes,no-recurrence-events
    |50-59,ge40,15-19,0-2,no,2,right,left_low,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,1,right,left_up,no,no-recurrence-events
    |30-39,premeno,15-19,6-8,yes,3,left,left_low,yes,recurrence-events
    |50-59,ge40,20-24,3-5,yes,2,right,left_up,no,no-recurrence-events
    |50-59,ge40,10-14,0-2,no,2,right,left_low,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,30-34,3-5,yes,3,left,left_low,no,no-recurrence-events
    |40-49,premeno,15-19,15-17,yes,3,left,left_low,no,recurrence-events
    |60-69,ge40,30-34,0-2,no,3,right,central,no,recurrence-events
    |60-69,ge40,25-29,3-5,?,1,right,left_low,yes,no-recurrence-events
    |50-59,ge40,25-29,0-2,no,3,left,right_up,no,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,3,right,left_up,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,1,left,left_low,yes,recurrence-events
    |30-39,premeno,15-19,0-2,no,1,left,left_low,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,2,right,left_up,no,no-recurrence-events
    |60-69,ge40,45-49,6-8,yes,3,left,central,no,no-recurrence-events
    |40-49,ge40,20-24,0-2,no,3,left,left_low,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,1,right,right_low,no,no-recurrence-events
    |30-39,premeno,35-39,0-2,no,3,left,left_low,no,recurrence-events
    |40-49,premeno,35-39,9-11,yes,2,right,right_up,yes,no-recurrence-events
    |60-69,ge40,25-29,0-2,no,2,right,left_low,no,no-recurrence-events
    |50-59,ge40,20-24,3-5,yes,3,right,right_up,no,recurrence-events
    |30-39,premeno,15-19,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,premeno,30-34,0-2,no,3,left,right_up,no,recurrence-events
    |60-69,ge40,10-14,0-2,no,2,right,left_up,yes,no-recurrence-events
    |40-49,premeno,35-39,0-2,yes,3,right,left_up,yes,no-recurrence-events
    |50-59,premeno,50-54,0-2,yes,2,right,left_up,yes,no-recurrence-events
    |50-59,ge40,40-44,0-2,no,3,right,left_up,no,no-recurrence-events
    |70-79,ge40,15-19,9-11,?,1,left,left_low,yes,recurrence-events
    |50-59,lt40,30-34,0-2,no,3,right,left_up,no,no-recurrence-events
    |40-49,premeno,0-4,0-2,no,3,left,central,no,no-recurrence-events
    |70-79,ge40,40-44,0-2,no,1,right,right_up,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,?,2,left,right_low,yes,no-recurrence-events
    |50-59,ge40,25-29,15-17,yes,3,right,left_up,no,no-recurrence-events
    |50-59,premeno,20-24,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,ge40,35-39,15-17,no,3,left,left_low,no,no-recurrence-events
    |50-59,ge40,50-54,0-2,no,1,right,right_up,no,no-recurrence-events
    |30-39,premeno,0-4,0-2,no,2,right,central,no,recurrence-events
    |50-59,ge40,40-44,6-8,yes,3,left,left_low,yes,recurrence-events
    |40-49,premeno,30-34,0-2,no,2,right,right_up,yes,no-recurrence-events
    |40-49,ge40,20-24,0-2,no,3,left,left_up,no,no-recurrence-events
    |40-49,premeno,30-34,15-17,yes,3,left,left_low,no,recurrence-events
    |40-49,ge40,20-24,0-2,no,2,right,left_up,no,recurrence-events
    |50-59,ge40,15-19,0-2,no,1,right,central,no,no-recurrence-events
    |30-39,premeno,25-29,0-2,no,2,right,left_low,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,2,left,left_low,no,no-recurrence-events
    |50-59,premeno,50-54,9-11,yes,2,right,left_up,no,recurrence-events
    |30-39,premeno,10-14,0-2,no,1,right,left_low,no,no-recurrence-events
    |50-59,premeno,25-29,3-5,yes,3,left,left_low,yes,recurrence-events
    |60-69,ge40,25-29,3-5,?,1,right,left_up,yes,no-recurrence-events
    |60-69,ge40,10-14,0-2,no,1,right,left_low,no,no-recurrence-events
    |50-59,ge40,30-34,6-8,yes,3,left,right_low,no,recurrence-events
    |30-39,premeno,25-29,6-8,yes,3,left,right_low,yes,recurrence-events
    |50-59,ge40,10-14,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,premeno,15-19,0-2,no,1,left,left_low,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,2,right,central,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,3,left,right_up,no,recurrence-events
    |60-69,ge40,30-34,6-8,yes,2,right,right_up,no,no-recurrence-events
    |50-59,lt40,15-19,0-2,no,2,left,left_low,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,2,right,left_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,2,left,left_up,yes,no-recurrence-events
    |30-39,premeno,0-4,0-2,no,2,right,central,no,no-recurrence-events
    |50-59,ge40,35-39,0-2,no,3,left,left_up,no,no-recurrence-events
    |40-49,premeno,40-44,0-2,no,1,right,left_up,no,no-recurrence-events
    |30-39,premeno,25-29,6-8,yes,2,right,left_up,yes,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,1,right,left_low,no,no-recurrence-events
    |50-59,ge40,30-34,0-2,no,1,left,left_up,no,no-recurrence-events
    |60-69,ge40,20-24,0-2,no,1,right,left_up,no,recurrence-events
    |30-39,premeno,30-34,3-5,no,3,right,left_up,yes,recurrence-events
    |50-59,lt40,20-24,0-2,?,1,left,left_up,no,recurrence-events
    |50-59,premeno,10-14,0-2,no,2,right,left_up,no,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,2,right,left_up,no,no-recurrence-events
    |40-49,premeno,45-49,0-2,no,2,left,left_low,yes,no-recurrence-events
    |30-39,premeno,40-44,0-2,no,1,left,left_up,no,recurrence-events
    |50-59,premeno,10-14,0-2,no,1,left,left_low,no,no-recurrence-events
    |60-69,ge40,30-34,0-2,no,3,right,left_up,yes,recurrence-events
    |40-49,premeno,35-39,0-2,no,1,right,left_up,no,recurrence-events
    |40-49,premeno,20-24,3-5,yes,2,left,left_low,yes,recurrence-events
    |50-59,premeno,15-19,0-2,no,2,left,left_low,no,recurrence-events
    |50-59,ge40,30-34,0-2,no,3,right,left_low,no,no-recurrence-events
    |60-69,ge40,20-24,0-2,no,2,left,left_up,no,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,1,left,right_low,no,no-recurrence-events
    |60-69,ge40,30-34,3-5,yes,2,left,central,yes,recurrence-events
    |60-69,ge40,20-24,3-5,no,2,left,left_low,yes,recurrence-events
    |50-59,premeno,25-29,0-2,no,2,left,right_up,no,recurrence-events
    |50-59,ge40,30-34,0-2,no,1,right,right_up,no,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,2,left,right_low,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,30-34,0-2,no,2,left,left_low,yes,no-recurrence-events
    |30-39,premeno,30-34,0-2,no,2,left,left_up,no,no-recurrence-events
    |30-39,premeno,40-44,3-5,no,3,right,right_up,yes,no-recurrence-events
    |60-69,ge40,5-9,0-2,no,1,left,central,no,no-recurrence-events
    |60-69,ge40,10-14,0-2,no,1,left,left_up,no,no-recurrence-events
    |40-49,premeno,30-34,6-8,yes,3,right,left_up,no,recurrence-events
    |60-69,ge40,10-14,0-2,no,1,left,left_up,no,no-recurrence-events
    |40-49,premeno,35-39,9-11,yes,2,right,left_up,yes,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,1,right,left_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,yes,3,right,right_up,no,recurrence-events
    |50-59,premeno,25-29,0-2,yes,2,left,left_up,no,no-recurrence-events
    |40-49,premeno,15-19,0-2,no,2,left,left_low,no,no-recurrence-events
    |30-39,premeno,35-39,9-11,yes,3,left,left_low,no,recurrence-events
    |30-39,premeno,10-14,0-2,no,2,left,right_low,no,no-recurrence-events
    |50-59,ge40,30-34,0-2,no,1,right,left_low,no,no-recurrence-events
    |60-69,ge40,30-34,0-2,no,2,left,left_up,no,no-recurrence-events
    |60-69,ge40,25-29,0-2,no,2,left,left_low,no,no-recurrence-events
    |40-49,premeno,15-19,0-2,no,2,left,left_up,no,recurrence-events
    |60-69,ge40,15-19,0-2,no,2,right,left_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,2,left,right_low,no,no-recurrence-events
    |20-29,premeno,35-39,0-2,no,2,right,right_up,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,3,right,right_up,no,recurrence-events
    |40-49,premeno,25-29,0-2,no,2,right,left_low,no,recurrence-events
    |30-39,premeno,30-34,0-2,no,3,left,left_low,no,no-recurrence-events
    |30-39,premeno,15-19,0-2,no,1,right,left_low,no,recurrence-events
    |50-59,ge40,0-4,0-2,no,1,right,central,no,no-recurrence-events
    |50-59,ge40,0-4,0-2,no,1,left,left_low,no,no-recurrence-events
    |60-69,ge40,50-54,0-2,no,3,right,left_up,no,recurrence-events
    |50-59,premeno,30-34,0-2,no,1,left,central,no,no-recurrence-events
    |60-69,ge40,20-24,24-26,yes,3,left,left_low,yes,recurrence-events
    |40-49,premeno,25-29,0-2,no,2,left,left_up,no,no-recurrence-events
    |40-49,premeno,30-34,3-5,no,2,right,left_up,no,recurrence-events
    |50-59,premeno,20-24,3-5,yes,2,left,left_low,no,no-recurrence-events
    |50-59,ge40,15-19,0-2,yes,2,left,central,yes,no-recurrence-events
    |50-59,premeno,10-14,0-2,no,3,left,left_low,no,no-recurrence-events
    |30-39,premeno,30-34,9-11,no,2,right,left_up,yes,recurrence-events
    |60-69,ge40,10-14,0-2,no,1,left,left_low,no,no-recurrence-events
    |40-49,premeno,40-44,0-2,no,2,right,left_low,no,no-recurrence-events
    |50-59,ge40,30-34,9-11,?,3,left,left_up,yes,no-recurrence-events
    |40-49,premeno,50-54,0-2,no,2,right,left_low,yes,recurrence-events
    |50-59,ge40,15-19,0-2,no,2,right,right_up,no,no-recurrence-events
    |50-59,ge40,40-44,3-5,yes,2,left,left_low,no,no-recurrence-events
    |30-39,premeno,25-29,3-5,yes,3,left,left_low,yes,recurrence-events
    |60-69,ge40,10-14,0-2,no,2,left,left_low,no,no-recurrence-events
    |60-69,lt40,10-14,0-2,no,1,left,right_up,no,no-recurrence-events
    |30-39,premeno,30-34,0-2,no,2,left,left_up,no,recurrence-events
    |30-39,premeno,20-24,3-5,yes,2,left,left_low,no,recurrence-events
    |50-59,ge40,10-14,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,25-29,0-2,no,3,right,left_up,no,no-recurrence-events
    |50-59,ge40,25-29,3-5,yes,3,right,left_up,no,no-recurrence-events
    |40-49,premeno,30-34,6-8,no,2,left,left_up,no,no-recurrence-events
    |60-69,ge40,50-54,0-2,no,2,left,left_low,no,no-recurrence-events
    |50-59,premeno,30-34,0-2,no,3,left,left_low,no,no-recurrence-events
    |40-49,ge40,20-24,3-5,no,3,right,left_low,yes,recurrence-events
    |50-59,ge40,30-34,6-8,yes,2,left,right_low,yes,recurrence-events
    |60-69,ge40,25-29,3-5,no,2,right,right_up,no,recurrence-events
    |40-49,premeno,20-24,0-2,no,2,left,central,no,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,2,left,left_up,no,no-recurrence-events
    |40-49,premeno,50-54,0-2,no,2,left,left_low,no,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,2,right,central,no,recurrence-events
    |50-59,ge40,30-34,3-5,no,3,right,left_up,no,recurrence-events
    |40-49,ge40,25-29,0-2,no,2,left,left_low,no,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,1,right,left_up,no,recurrence-events
    |40-49,premeno,40-44,3-5,yes,3,right,left_up,yes,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,2,right,left_up,no,no-recurrence-events
    |40-49,premeno,20-24,3-5,no,2,right,left_up,no,no-recurrence-events
    |40-49,premeno,25-29,9-11,yes,3,right,left_up,no,recurrence-events
    |40-49,premeno,25-29,0-2,no,2,right,left_low,no,recurrence-events
    |40-49,premeno,20-24,0-2,no,1,right,right_up,no,no-recurrence-events
    |30-39,premeno,40-44,0-2,no,2,right,right_up,no,no-recurrence-events
    |60-69,ge40,10-14,6-8,yes,3,left,left_up,yes,recurrence-events
    |40-49,premeno,35-39,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,ge40,30-34,3-5,no,3,left,left_low,no,recurrence-events
    |40-49,premeno,5-9,0-2,no,1,left,left_low,yes,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,1,left,right_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,3,right,right_up,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,3,left,left_up,no,recurrence-events
    |50-59,ge40,5-9,0-2,no,2,right,right_up,no,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,2,right,right_low,no,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,2,left,right_up,no,recurrence-events
    |40-49,premeno,10-14,0-2,no,2,left,left_low,yes,no-recurrence-events
    |60-69,ge40,35-39,6-8,yes,3,left,left_low,no,recurrence-events
    |60-69,ge40,50-54,0-2,no,2,right,left_up,yes,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,2,right,left_up,no,no-recurrence-events
    |30-39,premeno,20-24,3-5,no,2,right,central,no,no-recurrence-events
    |30-39,premeno,30-34,0-2,no,1,right,left_up,no,recurrence-events
    |60-69,lt40,30-34,0-2,no,1,left,left_low,no,no-recurrence-events
    |40-49,premeno,15-19,12-14,no,3,right,right_low,yes,no-recurrence-events
    |60-69,ge40,20-24,0-2,no,3,right,left_low,no,recurrence-events
    |30-39,premeno,5-9,0-2,no,2,left,right_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,3,left,left_up,no,no-recurrence-events
    |60-69,ge40,30-34,0-2,no,3,left,left_low,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,1,right,right_low,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,1,left,right_low,no,no-recurrence-events
    |60-69,ge40,40-44,3-5,yes,3,right,left_low,no,recurrence-events
    |50-59,ge40,25-29,0-2,no,2,left,left_low,no,no-recurrence-events
    |50-59,premeno,30-34,0-2,no,3,right,left_up,yes,recurrence-events
    |40-49,ge40,30-34,3-5,no,3,left,left_low,no,recurrence-events
    |40-49,premeno,25-29,0-2,no,1,right,left_low,yes,no-recurrence-events
    |40-49,ge40,25-29,12-14,yes,3,left,right_low,yes,recurrence-events
    |40-49,premeno,40-44,0-2,no,1,left,left_low,no,recurrence-events
    |40-49,premeno,20-24,0-2,no,2,left,left_low,no,no-recurrence-events
    |50-59,ge40,25-29,0-2,no,1,left,right_low,no,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,2,right,left_up,no,no-recurrence-events
    |70-79,ge40,40-44,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,25-29,0-2,no,3,left,left_up,no,recurrence-events
    |50-59,premeno,25-29,0-2,no,2,left,left_low,no,no-recurrence-events
    |60-69,ge40,45-49,0-2,no,1,right,right_up,yes,recurrence-events
    |50-59,ge40,20-24,0-2,yes,2,right,left_up,no,no-recurrence-events
    |50-59,ge40,25-29,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,3,left,left_up,no,no-recurrence-events
    |40-49,premeno,20-24,3-5,no,2,right,left_low,no,no-recurrence-events
    |50-59,ge40,35-39,0-2,no,2,left,left_up,no,no-recurrence-events
    |30-39,premeno,20-24,0-2,no,3,left,left_up,yes,recurrence-events
    |60-69,ge40,30-34,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,25-29,0-2,no,3,right,left_low,no,no-recurrence-events
    |40-49,ge40,30-34,0-2,no,2,left,left_up,yes,no-recurrence-events
    |30-39,premeno,25-29,0-2,no,2,left,left_low,no,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,2,left,left_low,no,recurrence-events
    |30-39,premeno,20-24,0-2,no,2,left,right_low,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,2,right,left_low,no,no-recurrence-events
    |50-59,premeno,15-19,0-2,no,2,right,right_low,no,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,1,right,left_up,no,no-recurrence-events
    |60-69,ge40,20-24,0-2,no,2,right,left_up,no,no-recurrence-events
    |60-69,ge40,40-44,0-2,no,2,right,left_low,no,recurrence-events
    |30-39,lt40,15-19,0-2,no,3,right,left_up,no,no-recurrence-events
    |40-49,premeno,30-34,12-14,yes,3,left,left_up,yes,recurrence-events
    |60-69,ge40,30-34,0-2,yes,2,right,right_up,yes,recurrence-events
    |50-59,ge40,40-44,6-8,yes,3,left,left_low,yes,recurrence-events
    |50-59,ge40,30-34,0-2,no,3,left,?,no,recurrence-events
    |70-79,ge40,10-14,0-2,no,2,left,central,no,no-recurrence-events
    |30-39,premeno,40-44,0-2,no,2,left,left_low,yes,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,2,right,right_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,1,left,left_low,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,2,left,left_low,no,no-recurrence-events
    |40-49,premeno,10-14,0-2,no,2,left,left_low,no,no-recurrence-events
    |60-69,ge40,20-24,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,ge40,10-14,0-2,no,1,left,left_up,no,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,ge40,30-34,9-11,yes,3,left,right_low,yes,recurrence-events
    |50-59,ge40,10-14,0-2,no,2,left,left_low,no,no-recurrence-events
    |40-49,premeno,30-34,0-2,no,1,left,right_up,no,no-recurrence-events
    |70-79,ge40,0-4,0-2,no,1,left,right_low,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,3,right,left_up,yes,no-recurrence-events
    |50-59,premeno,25-29,0-2,no,3,right,left_low,yes,recurrence-events
    |50-59,ge40,40-44,0-2,no,2,left,left_low,no,no-recurrence-events
    |60-69,ge40,25-29,0-2,no,3,left,right_low,yes,recurrence-events
    |40-49,premeno,30-34,3-5,yes,2,right,left_low,no,no-recurrence-events
    |50-59,ge40,20-24,0-2,no,2,left,left_up,no,recurrence-events
    |70-79,ge40,20-24,0-2,no,3,left,left_up,no,no-recurrence-events
    |30-39,premeno,25-29,0-2,no,1,left,central,no,no-recurrence-events
    |60-69,ge40,30-34,0-2,no,2,left,left_low,no,no-recurrence-events
    |40-49,premeno,20-24,3-5,yes,2,right,right_up,yes,recurrence-events
    |50-59,ge40,30-34,9-11,?,3,left,left_low,yes,no-recurrence-events
    |50-59,ge40,0-4,0-2,no,2,left,central,no,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,3,right,left_low,yes,no-recurrence-events
    |30-39,premeno,35-39,0-2,no,3,left,left_low,no,recurrence-events
    |60-69,ge40,30-34,0-2,no,1,left,left_up,no,no-recurrence-events
    |60-69,ge40,20-24,0-2,no,1,left,left_low,no,no-recurrence-events
    |50-59,ge40,25-29,6-8,no,3,left,left_low,yes,recurrence-events
    |50-59,premeno,35-39,15-17,yes,3,right,right_up,no,recurrence-events
    |30-39,premeno,20-24,3-5,yes,2,right,left_up,yes,no-recurrence-events
    |40-49,premeno,20-24,6-8,no,2,right,left_low,yes,no-recurrence-events
    |50-59,ge40,35-39,0-2,no,3,left,left_low,no,no-recurrence-events
    |50-59,premeno,35-39,0-2,no,2,right,left_up,no,no-recurrence-events
    |40-49,premeno,25-29,0-2,no,2,left,left_up,yes,no-recurrence-events
    |40-49,premeno,35-39,0-2,no,2,right,right_up,no,no-recurrence-events
    |50-59,premeno,30-34,3-5,yes,2,left,left_low,yes,no-recurrence-events
    |40-49,premeno,20-24,0-2,no,2,right,right_up,no,no-recurrence-events
    |60-69,ge40,15-19,0-2,no,3,right,left_up,yes,no-recurrence-events
    |50-59,ge40,30-34,6-8,yes,2,left,left_low,no,no-recurrence-events
    |50-59,premeno,25-29,3-5,yes,2,left,left_low,yes,no-recurrence-events
    |30-39,premeno,30-34,6-8,yes,2,right,right_up,no,no-recurrence-events
    |50-59,premeno,15-19,0-2,no,2,right,left_low,no,no-recurrence-events
    |50-59,ge40,40-44,0-2,no,3,left,right_up,no,no-recurrence-events
  """.stripMargin.trim.split("\n")

  /**
   * dataset playtennis
   */
  val strings =
    """
      |sunny,hot,high,FALSE,no
      |sunny,hot,high,TRUE,no
      |overcast,hot,high,FALSE,yes
      |rainy,mild,high,FALSE,yes
      |rainy,cool,normal,FALSE,yes
      |rainy,cool,normal,TRUE,no
      |overcast,cool,normal,TRUE,yes
      |sunny,mild,high,FALSE,no
      |sunny,cool,normal,FALSE,yes
      |rainy,mild,normal,FALSE,yes
      |sunny,mild,normal,TRUE,yes
      |overcast,mild,high,TRUE,yes
      |overcast,hot,normal,FALSE,yes
      |rainy,mild,high,TRUE,no
    """.stripMargin.trim.split("\n")




  val dataset = strings.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  val datasetBreastCancer = stringsBreastCancer.map { string =>
    string.split(",").zipWithIndex.map {
      case (value, index) =>
        (index, value)
    }
  }

  /**
   * Training Parameter
   */
  val parameter = DeepNetworkParameter(
    hiddenLayerSize = List(3,3,3,3,3,3,3),
    outputPerceptronSize = 2,
    targetClassPosition = -1,
    iteration = 100000,
    epsilon = 0.00000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  val parameterBreastCancer = DeepNetworkParameter(
    hiddenLayerSize = List(8,8,8,8,8),
    outputPerceptronSize = 2,
    targetClassPosition = -1,
    iteration = 100000,
    epsilon = 0.00000001,
    momentum = 0.75,
    learningRate = 0.5,
    synapsysFactory = RandomSynapsysFactory(),
    activationFunction = SigmoidFunction,
    inputPerceptronSize = dataset.head.length - 1
  )

  val targetClass = if(parameter.targetClassPosition == -1) dataset.head.length - 1 else parameter.targetClassPosition
  val targetClassBreastCancer = if(parameterBreastCancer.targetClassPosition == -1) datasetBreastCancer.head.length - 1 else parameterBreastCancer.targetClassPosition

  val finalDataSet = StandardNormalization.normalize(
    dataset.map(data => {
      data.map { case (index, value) =>
        priorKnowledge(index)(value)
      }
    }).toList
    , targetClass)

  val finalDataSetBreastCancer = StandardNormalization.normalize(
    datasetBreastCancer.map(data => {
      data.map { case (index, value) =>
        priorKnowledgeBreastCancer(index)(value)
      }
    }).toList
    , targetClassBreastCancer)


//  finalDataSet.foreach { array =>
//    println(array.mkString(","))
//  }
//
//  finalDataSetBreastCancer.foreach { array =>
//    println(array.mkString(","))
//  }


  var logger  = LoggerFactory.getLogger("Main Objects")

  test("traininig and classification and save model") {
    // training
    try {

//      val network = DeepNetworkAlgorithm.train(finalDataSetBreastCancer, parameterBreastCancer)
      val network = DeepNetworkAlgorithm.train(finalDataSet, parameter)

      val validator = DeepNetworkValidation()

//      val result = validator.classification(network, DeepNetworkClassification, finalDataSetBreastCancer, SigmoidFunction)
      val result = validator.classification(network, DeepNetworkClassification, finalDataSet, SigmoidFunction)
//      logger.info("result finding : "+ result.toString())

//      val validateResult = validator.validate(result, finalDataSetBreastCancer, 4)
      val validateResult = validator.validate(result, finalDataSet, 4)

      logger.info("after validation result : "+validateResult.toString())

      val accuration = validator.accuration(validateResult) {
        EitherThresholdFunction(0.7, 0.0, 1.0)
      }

      logger.info("after accuration counting : "+accuration.toString())

      // classification
//      finalDataSetBreastCancer.foreach { data =>
      finalDataSet.foreach { data =>
        val realScore = DeepNetworkClassification(data, network, SigmoidFunction)
        realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
          val percent = Math.round(p._1 * 100.0)
          val score = if (p._1 > 0.7) 1.0 else 0.0
          val originalClass = data(targetClass).asInstanceOf[BinaryValue].get(p._2)
          println(s"real $p== percent $percent% == score $score == targetClass ${originalClass}")
          assert(score == originalClass)
        })
      }

      // save model
      NetworkSerialization.save(network, new FileOutputStream(
        new File("target" + File.separator + "cuaca.json")))
    }catch {
      case npe : NullPointerException => npe.printStackTrace()
      case e : Exception => e.printStackTrace()
    }
  }

  test("load model and classification") {

    // load model
    val network = NetworkSerialization.load(inputStream = new FileInputStream(
      new File("target" + File.separator + "cuaca.json")), typeOfInference = "NeuralNet").asInstanceOf[Network]

    // classification
    finalDataSet.foreach { data =>
      val realScore = BasicClassification(data, network, SigmoidFunction)
      realScore.asInstanceOf[BinaryValue].get.zipWithIndex.foreach(p => {
        val percent = Math.round(p._1 * 100.0)
        val score = if (p._1 > 0.7) 1.0 else 0.0
        val originalClass = data(targetClass).asInstanceOf[BinaryValue].get(p._2)
        println(s"real $p== percent $percent% == score $score == targetClass ${originalClass}")
        assert(score == originalClass)
      })
    }
  }

}
