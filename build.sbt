organization := "com.github.rzykov"

name := "fastml4j"

version := "0.1.1-SNAPSHOT"

scalaVersion := "2.11.8"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "provided"

libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.1"

libraryDependencies += "com.typesafe" % "config" % "1.3.1"

val nd4jVersion = "0.9.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion

resolvers += "Maven" at "http://repo1.maven.org/maven2/"

fork in Test := true

val meta = """META.INF(.)*""".r

useGpg := true

homepage := Some(url("https://github.com/rzykov/fastml4j"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/rzykov/fastml4j"),
    "scm:git@github.com:rzykov/fastml4j.git"
  )
)

developers := List(
  Developer("rzykov",
    "Roman Zykov",
    "rzykov@gmail.com",
    url("https://github.com/rzykov")))

licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

pomIncludeRepository := (_ => false)

publishArtifact in Test := false

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

publishMavenStyle := true

scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Yno-adapted-args",
  "-Ywarn-dead-code",
  "-Ywarn-unused",
  "-Ywarn-value-discard"
)


