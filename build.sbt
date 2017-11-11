organization := "org.github.rzykov"

name := "fastml4j"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "provided"

libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.1"

libraryDependencies += "com.typesafe" % "config" % "1.3.1"

val nd4jVersion = "0.9.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion

resolvers += "Maven" at "http://repo1.maven.org/maven2/"

fork in Test := true

val meta = """META.INF(.)*""".r

scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Yno-adapted-args",
  "-Ywarn-dead-code",
  "-Ywarn-unused",
  "-Ywarn-value-discard"
)


