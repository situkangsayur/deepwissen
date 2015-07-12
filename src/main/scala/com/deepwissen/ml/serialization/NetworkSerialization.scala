/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.serialization

import java.io.{InputStream, OutputStream}

import com.deepwissen.ml.algorithm.{MarkovChain, InferencesNetwork, Network}
import org.apache.commons.io.IOUtils

/**
 * Utility class for save network model and load network model
 * @author Eko Khannedy
 * @since 2/25/15
 */
object NetworkSerialization {

  /**
   * Save network model to output stream
   * @param network network model
   * @param outputStream output stream
   * @param closeStream auto close output stream? default true
   */
  def save(network: InferencesNetwork, outputStream: OutputStream, closeStream: Boolean = true): Unit = {
    val model = if(network.isInstanceOf[Network]) NetworkModel(network.asInstanceOf[Network])
                else if(network.isInstanceOf[MarkovChainModel]) MarkovChainModel(network.asInstanceOf[MarkovChain])

    NetworkMapper.writeValue(outputStream, model)
    if (closeStream) IOUtils.closeQuietly(outputStream)
  }

  /**
   * Load network model from input stream
   * @param inputStream input stream
   * @param closeStream auto close input stream? default true
   * @return
   */
  def load(inputStream: InputStream, closeStream: Boolean = true, typeOfInference: String): InferencesNetwork = {
    val result = typeOfInference match {
      case "MarkovChain" =>
        val model = NetworkMapper.readValue(inputStream, classOf[MarkovChainModel])
        if (closeStream) IOUtils.closeQuietly(inputStream)
        MarkovChain(model.asInstanceOf[MarkovChainModel])
      case "NeuralNet" =>
        val model = NetworkMapper.readValue(inputStream, classOf[NetworkModel])
        if (closeStream) IOUtils.closeQuietly(inputStream)
        Network(model)
    }

    result
  }

}
