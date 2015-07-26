package com.deepwissen.ml.algorithm.test

import com.deepwissen.ml.algorithm.Autoencoder._
import com.deepwissen.ml.algorithm.{BackpropragationParameter, AutoencoderParameter, AbstractAutoEncoder}
import com.deepwissen.ml.algorithm.networks.{Network, AutoencoderNetwork}
import com.deepwissen.ml.utils.Denomination

import scala.annotation.tailrec

/**
 * Created by hendri_k on 7/26/15.
 */
object AutoencoderBPRun extends AutoencoderBP[List[Array[Denomination[_]]]] {
  /**
   * Run training with given dataset
   * @param dataset dataset
   * @param parameter training parameter
   * @return network model
   */
  override def doTrain(network: Network, dataset: List[Array[Denomination[_]]], parameter: BackpropragationParameter): Unit = {

    @tailrec
    def iterate(iteration: Int, error: Double): Unit = {
      if (error < parameter.epsilon || iteration > parameter.iteration) {
        // stop iteration
      } else {
        // print information
        println(s"###### error : $error : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
        // run training
        val trainError = dataset.foldLeft(0.0)((value, data) => value + doTrainData(data, network, parameter))
        // next iteration
        iterate(iteration + 1, trainError / dataset.length)
      }
    }

    // start first iteration with given max error
    iterate(1, Double.MaxValue)
  }
}
