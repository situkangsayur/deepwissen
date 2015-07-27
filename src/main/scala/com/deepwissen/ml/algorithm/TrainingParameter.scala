/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.algorithm

import com.deepwissen.ml.function.{UniformDistribution, Sampling, SigmoidFunction, ActivationFunction}

/**
 * Base trait for training parameter
 * @author Eko Khannedy
 * @since 2/25/15
 */
trait TrainingParameter

/**
 * Parameter for backpropragation algorithm
 * @param inputPerceptronSize total perceptron in input layer
 * @param hiddenLayerSize total hidden layer in network model default is 1
 * @param synapsysFactory synapsys factory default is random synapsys factory
 * @param activationFunction activation function default is sigmoid function
 * @param learningRate learning rate default is 0.5
 * @param momentum momentum default is 0.75
 * @param epsilon epsilon default is 0.000001
 * @param iteration total maximum iteration, default is Int.MaxValue
 */
case class BackpropragationParameter(inputPerceptronSize: Int,
                                     hiddenLayerSize: Int = 1,
                                     outputPerceptronSize: Int = 1,
                                     targetClassPosition: Int = -1,
                                     synapsysFactory: SynapsysFactory[_] = RandomSynapsysFactory(),
                                     activationFunction: ActivationFunction = SigmoidFunction,
                                     learningRate: Double = 0.5,
                                     momentum: Double = 0.75,
                                     epsilon: Double = 0.000001,
                                     iteration: Int = Int.MaxValue) extends TrainingParameter


case class AutoencoderParameter(inputPerceptronSize: Int,
                                hiddenPerceptronSize: Int,
                                synapsysFactory: SynapsysFactory[_] = RandomSynapsysFactory(),
                                targetClassPosition : Int = -1,
                                activationFunction: ActivationFunction = SigmoidFunction,
                                learningRate: Double = 0.5,
                                momentum: Double = 0.75,
                                epsilon: Double = 0.000001,
                                iteration: Int = Int.MaxValue) extends TrainingParameter


case class DeepNetworkParameter(inputPerceptronSize: Int,
                                hiddenLayerSize: List[Int] = List(4),
                                outputPerceptronSize: Int = 1,
                                targetClassPosition: Int = -1,
                                synapsysFactory: SynapsysFactory[_] = RandomSynapsysFactory(),
                                activationFunction: ActivationFunction = SigmoidFunction,
                                learningRate: Double = 0.5,
                                momentum: Double = 0.75,
                                epsilon: Double = 0.000001,
                                iteration: Int = Int.MaxValue) extends TrainingParameter


//case class RbmParamter(inputPerceptronSize: Int,
//                       hiddenLayerSize: Int = 1,
//                       gibbsStep: Int = 3000,
//                       targetClassPosition: Int = -1,
//                       synapsysFactory: SynapsysFactory[_] = RandomSynapsysFactory(),
//                       activationFunction: ActivationFunction = SigmoidFunction,
//                       learningRate: Double = 0.75,
//                       epsilon: Double = 0.000001,
//                       iteration: Int = Int.MaxValue,
//                       sampling : Sampling = UniformDistribution) extends TrainingParameter

case class DbnParamter(inputPerceptronSize: Int,
                       hiddenLayerSize: Int = 1,
                       outputPerceptronSize: Int = 1,
                       gibbsStep: Int = 3000,
                       targetClassPosition: Int = -1,
                       synapsysFactory: SynapsysFactory[_] = RandomSynapsysFactory(),
                       activationFunction: ActivationFunction = SigmoidFunction,
                       learningRate: Double = 0.75,
                       epsilon: Double = 0.000001,
                       iteration: Int = Int.MaxValue) extends TrainingParameter

case class GibbsParameter(inputPerceptronSize: Int,
                          hiddenPerceptronSize: Int = 3,
                          k : Int,
                          targetClassPosition: Int = -1,
                          synapsysFactory: SynapsysFactory[_] = RandomSynapsysFactory(),
                          activationFunction: ActivationFunction = SigmoidFunction,
                          learningRate: Double = 0.75,
                          momentum: Double = 0.75,
                          epsilon: Double = 0.000001,
                          iteration: Int = Int.MaxValue,
                          sampling : Sampling = UniformDistribution(0,1),
                          dataSize : Int) extends TrainingParameter