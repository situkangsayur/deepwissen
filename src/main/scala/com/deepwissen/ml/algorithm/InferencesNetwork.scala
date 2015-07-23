package com.deepwissen.ml.algorithm

/**
 * Created by hendri_k on 7/13/15.
 */
trait InferencesNetwork {
//  @transient
//  private val allPerceptrons: Map[String, Perceptron]

  /**
   * Get perceptron by id
   * @param perceptronId perceptron id
   * @return Perceptron
   */
  def getPerceptron(perceptronId: String): Perceptron

//  @transient
//  private val synapsiesLookupPair: Map[(String, String), Synapsys]

  /**
   * Get synapsys that go from source perceptron to target perceptron
   * @param fromPerceptronId source perceptron
   * @param toPerceptronId target perceptron
   * @return synapsys
   */
  def getSynapsys(fromPerceptronId: String, toPerceptronId: String) : Synapsys

//  @transient
//  private val synapsiesLookupFrom: Map[String, Seq[Synapsys]]

  /**
   * Get all synapsies that go from source perceptron
   * @param perceptronId source perceptron
   * @return list of synapsies
   */
  def getSynapsiesFrom(perceptronId: String) : Seq[Synapsys]

//  @transient
//  private val synapsiesLookupTo: Map[String, Seq[Synapsys]] = synapsies.groupBy(_.to.id)

  /**
   * Get all synapsies that go to target perceptron
   * @param perceptronId target perceptron
   * @return list of synapsies
   */
  def getSynapsiesTo(perceptronId: String) : Seq[Synapsys]

  /**
   * Get perceptron weight, calculate from all synapsies and perceptron source
   * @param perceptron perceptron
   * @return weight
   */
  def getPerceptronWeightTo(perceptron: Perceptron): Double


  /**
   * Get perceptron weight, calculate from all synapsies and perceptron out source
   * @param perceptron
   * @return
   */
  def getPerceptronWeightFrom(perceptron: Perceptron): Double

}
