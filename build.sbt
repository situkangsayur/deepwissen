name := """machine-learning"""

organization := "com.deepwissen"

version := "1.0"

scalaVersion := "2.11.6"

// Change this to another test framework if you prefer
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

// Uncomment to use Akka
//libraryDependencies += "com.typesafe.akka" % "akka-actor_2.11" % "2.3.9"

//libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.0.9"
libraryDependencies += "org.slf4j" % "slf4j-api" % "1.6.4"

libraryDependencies += "org.specs2" %% "specs2-core" % "2.4.15" % "test"

libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.6.0"

libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.5.1"

libraryDependencies += "com.fasterxml.jackson.core" % "jackson-databind" % "2.5.1"

libraryDependencies += "com.fasterxml.jackson.core" % "jackson-annotations" % "2.5.1"

libraryDependencies += "com.fasterxml.jackson.module" % "jackson-module-scala_2.11" % "2.5.1"

libraryDependencies += "org.apache.hbase" % "hbase-client" % "1.0.1"

libraryDependencies += "org.apache.hbase" % "hbase-common" % "1.0.1"

libraryDependencies += "org.apache.hbase" % "hbase-annotations" % "1.0.1"

libraryDependencies += "redis.clients" % "jedis" % "2.7.2"

libraryDependencies += "org.mongodb" %% "casbah" % "2.8.1"