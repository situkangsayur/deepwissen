package com.deepwissen.ml.algorithm

import java.io.{FileOutputStream, PrintStream}

import com.deepwissen.ml.algorithm.BasicBackpropagation._
import com.deepwissen.ml.algorithm.networks.{DeepNetwork, Network}
import com.deepwissen.ml.utils.Denomination

import scala.annotation.tailrec

/**
 * Created by hendri_k on 7/28/15.
 */
object DeepNetworkAlgorithm extends AbstractDeepNetwork {

  val file = new PrintStream(new FileOutputStream("result-ds1-dnn1-2007.txt"), true)
  /**
   * Run training with given dataset
   * @param dataset dataset
   * @param parameter training parameter
   * @return network model
   */
  override def doTrain(network: DeepNetwork, dataset: List[Array[Denomination[_]]], parameter: DeepNetworkParameter): Unit = {

    @tailrec
    def iterate(iteration: Int, error: Double): Unit = {
//      if (error < parameter.epsilon) {
        if (error < parameter.epsilon || iteration > parameter.iteration) {
        // stop iteration
      } else {
        val rmse = math.sqrt(error)
        // print information

        if(iteration % 100 == 0){
          //          logger.info(s"###### MSE : $error : RMSE : $rmse : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
          file.append(s"###### MSE : $error : RMSE : $rmse : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
        }

        println(s"###### MSE : $error : RMSE : $rmse : iteration :$iteration ---> max it ${parameter.iteration} max ep ${parameter.epsilon}")
        // run training
        val trainError = dataset.foldLeft(0.0)((value, data) => value + doTrainData(data, network, parameter))
        // next iteration
        iterate(iteration + 1, trainError / (2 * dataset.length * parameter.outputPerceptronSize))
      }
    }

    // start first iteration with given max error
    iterate(1, Double.MaxValue)
    file.close()
  }
}
