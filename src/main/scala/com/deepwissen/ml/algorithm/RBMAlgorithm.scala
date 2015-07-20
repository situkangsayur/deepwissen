package com.deepwissen.ml.algorithm

import com.deepwissen.ml.utils.Denomination

import scala.annotation.tailrec

/**
 * Created by hendri_k on 7/17/15.
 */
object RBMAlgorithm extends AbstractRestrictedBoltzmannMachine[List[Array[Denomination[_]]]]{

    /**
     * Run training with given dataset
     * @param dataset dataset
     * @param parameter training parameter
     * @return network model
     */
    override def doTrain(network: MarkovChain, dataset: List[Array[Denomination[_]]], parameter: GibbsParameter): Unit = {

      @tailrec
      def iterate(iteration: Int, error: Double): Unit = {
        if (error < parameter.epsilon || iteration > parameter.iteration) {
          // stop iteration
        } else {
          println("##########################################################################################################################")
          // run training
          val listOfPartialFreeEnergy = dataset.map( data => doTrainData(data, network, parameter))
//          val partialFreeEnergy = dataset.foldLeft(0.0)((value, data) => value + doTrainData(data, network, parameter))

          val partialFunction = listOfPartialFreeEnergy.foldLeft(0.0)((temp, value) => temp + value)
          val lostFunction = listOfPartialFreeEnergy.foldLeft(0.0)((temp, value) => temp + (value/partialFunction))/listOfPartialFreeEnergy.size
          // next iteration
          // print information
          println(s"###### error : $error : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
          println("--------------------------------------------------------------------------------------------------------------------------")

          iterate(iteration + 1, lostFunction)
        }
      }

      // start first iteration with given max error
      iterate(1, Double.MaxValue)
    }

}
