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

      val tempDataset = dataset.zipWithIndex

      @tailrec
      def iterate(iteration: Int, error: Double, resultParam: List[List[Perceptron]]): Unit = {
        if (error < parameter.epsilon || iteration > parameter.iteration) {
          // stop iteration
        } else {
          println("##########################################################################################################################")
          // run training
          val tempResult: List[(Double, List[Perceptron])] =
                tempDataset.map( data => doTrainData(data._1,if(iteration == 1) null else resultParam(data._2),network, parameter))

          val listOfPartialFreeEnergy = tempResult.map(x => x._1)

          val z = listOfPartialFreeEnergy.foldLeft(0.0)((temp, value) => temp + value)
          val lostFunction = listOfPartialFreeEnergy.foldLeft(0.0)((temp, value) => {
            println("p(x) = "+value + " / " z)
            temp + (value/z)})/listOfPartialFreeEnergy.size
          // next iteration
          // print information
          println(s"###### error : $error : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
          println("--------------------------------------------------------------------------------------------------------------------------")

          iterate(iteration + 1, lostFunction, tempResult.map(x => x._2))
        }
      }

      // start first iteration with given max error
      iterate(1, Double.MaxValue, null)
    }

}
