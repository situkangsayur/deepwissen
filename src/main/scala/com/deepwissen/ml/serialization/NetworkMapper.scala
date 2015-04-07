/*
 * Copyright (c) 2015, DeepWissen and/or its affiliates. All rights reserved.
 * DEEPWISSEN PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

package com.deepwissen.ml.serialization

import java.text.SimpleDateFormat

import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.databind.{DeserializationFeature, ObjectMapper, PropertyNamingStrategy, SerializationFeature}
import com.fasterxml.jackson.module.scala.DefaultScalaModule

/**
 * Jackson mapper for Network Model
 *
 * @author Eko Khannedy
 * @since 2/25/15
 */
object NetworkMapper extends ObjectMapper {

  setDateFormat(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"))
  setPropertyNamingStrategy(PropertyNamingStrategy.CAMEL_CASE_TO_LOWER_CASE_WITH_UNDERSCORES)
  enable(SerializationFeature.INDENT_OUTPUT)
  configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
  configure(DeserializationFeature.FAIL_ON_NUMBERS_FOR_ENUMS, false)
  setSerializationInclusion(JsonInclude.Include.NON_NULL)
  registerModule(DefaultScalaModule)

}
