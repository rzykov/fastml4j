import AssemblyKeys._

assemblySettings

organization := "ru.retailrocket.spark"

name := "fastml4j"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "provided"

libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.1"

libraryDependencies += "com.typesafe" % "config" % "1.3.1"

val nd4jVersion = "0.8.0"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion

libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion

resolvers += "Maven" at "http://repo1.maven.org/maven2/"

assembleArtifact in packageScala := false

fork in Test := true

val meta = """META.INF(.)*""".r

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
  {
    case PathList("javax", "servlet", xs @ _*) => MergeStrategy.last
    case PathList("javax", "activation", xs @ _*) => MergeStrategy.last
    case PathList("org", "apache", xs @ _*) => MergeStrategy.last
    case PathList("com", "esotericsoftware", xs @ _*) => MergeStrategy.last
    case PathList("plugin.properties") => MergeStrategy.last
    case meta(_) => MergeStrategy.discard
    case x => old(x)
  }
}

