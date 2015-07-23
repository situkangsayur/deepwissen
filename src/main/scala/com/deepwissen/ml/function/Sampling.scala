package com.deepwissen.ml.function

import scala.util.Random


/**
 * Created by hendri_k on 7/24/15.
 */
trait Sampling {

  /**
   * get value of sampling using a distribution
   * @param value
   * @return
   */
  def getValue(value : Double) : Double

}


/**
 * Unifor Distribution to get binomial distribution
 * @param min
 * @param max
 */
case class UniformDistribution(min : Double, max : Double, random : Random = new Random(123)) extends Sampling {

  def uniform : Double = random.nextDouble() * (max - min) + min

  override def getValue(value: Double): Double = {
    if(value < 0.0 || value > 1.0) 0.0
    else if(value > random.nextDouble()) 1.0 else 0.0
  }
}

