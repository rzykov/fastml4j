# Fastml4j
Fast Scala and nd4j based machine learning experimental framework. It's based on 
[ND4J](https://github.com/deeplearning4j/nd4j), the library for a scientific computing on the JVM. 

## Goal
My goal is to create simple and fast  machine learning library for Scala and, if possible, 
for JVM.  

I work heavily with the Spark's MLLib and after that I want to say: Don't use the distributed machine 
learning algorithms until you really need it. We prefer to prepare datasets 
in the distributed manner but 
prefer to use non-distributed ML libraries due to performance reasons. I saw a case when 
MLLib's  Random forests took 2 hours to train and the same method from the Smile library 
took only 5 minutes. You'll always pay for the distributed ML. 

Unfortunately, there is a lack of good machine learning libraries for Scala. I found a very 
good library: [Smile](https://github.com/haifengl/smile). It's been written in Java, fast. But it doesn't use a vectorwised 
approach. The vectorwise approach makes code more readable and faster. You can use matrix maths 
operations with it with only a few symbols.

## Features
* Logistic regression
* Linear SVM
* Linear regression (classical OLS)
* Binary classification metrics
* Regression metrics

## Roadmap
* Decision trees with categorical variables and missing data
* General ensembles
* Random forests
* Advanced optimising algorithms
* Gradient boosted trees. I'm going to use the existed library [Catboost](https://github.com/catboost/catboost) 
via [JavaCPP](https://github.com/bytedeco/javacpp-presets). It's hard to write own good GBRT.

## Getting started
* Build:
```bash
git clone https://github.com/rzykov/fastml4j
cd fastml4j
sbt package
```   
* [Scala Doc for this package](https://rzykov.github.io/fastml4j/api/)
* Link to Scala notebook
* How to use it from maven
```
Apache Maven
<dependency>
    <groupId>com.github.rzykov</groupId>
    <artifactId>fastml4j_2.11</artifactId>
    <version>0.1</version>
</dependency>
    
Apache Buildr
'com.github.rzykov:fastml4j_2.11:jar:0.1'
   
Apache Ivy
<dependency org="com.github.rzykov" name="fastml4j_2.11" rev="0.1" />
    
Groovy Grape
@Grapes( 
@Grab(group='com.github.rzykov', module='fastml4j_2.11', version='0.1') 
)
    
Gradle/Grails
compile 'com.github.rzykov:fastml4j_2.11:0.1'
    
Scala SBT
libraryDependencies += "com.github.rzykov" % "fastml4j_2.11" % "0.1"
    
Leiningen
[com.github.rzykov/fastml4j_2.11 "0.1"]
```

## FAQ
* __Why did you choose ND4J?__
  It's supports many platforms and  NVIDIA CUDA GPU out of the box. 
* __Why did you write own implicits instead of using nd4s library?__
  Nd4s looks too complicated for my purposes. Also found that it doesn't contain some DSL 
  elements. Personally, I don't like Implicits, but in this case they are in the right place.
  My preferable way of using them is to import them explicitly in any source file. 
  It gives a hint to the reader to show up the fact for using implicits and where to find them.  
* __Why did you choose Float rather than Double?__
  ND4J uses Float as a default type. It looks reasonable  because it saves a memory. 
  I also want to conduct some experiments with the Raspberry Pie in future which have 1 GB of RAM only.
 
## Contributions
I'm interesting in a code review of this project and getting a feedback from users.

## Why Scala for machine learning?

It's biased to Spark but also works for the general data science.

It's very important to choose a right tool for a data analysis. There are lots of questions like "What Machine Learning tool is better?" at Kaggle.com's forums. Top ranks are occupied by R and Python. I will try to write about my migration to the Scala/Spark technology stack.

We do machine learning tasks at a very large scale in Retailrocket.net. Previously we used  IPython + Pyhs2 (hive driver in Python) + Pandas + Sklearn for building prototypes. We decided to move to Apache Spark at the end of summer of 2014 because our experiments with this one showed 3-4x raising of performance on the same cluster.

At that time, four languages  were used simultaneously: Hive, Pig, Java, Python. Sometimes we'd got very serious problems with this "Zoo". With Spark, we could use only one programming language for prototypes and a code in production. This is a great benefit for our small team.

Spark supports Python/Scala/Java via API. Spark is written in Scala hence, it was chosen as the main development language by our team. We can analyse the source code of Spark and make patches for it. Also, Scala is a JVM (Java) based language. It's very important because Hadoop was written in Java.

Scala:
* (+) functional; nice for data scientists
* (+) native for Spark; important, if you want to learn Spark internals
* (+) based on JVM; so it's native for Hadoop
* (+) strong static types; get errors at the compilation stage
* (-) hard to learn; 

Python:
* (+) popular;
* (+) simple;
* (-) dynamic typing;
* (-) performance is worse than Scala.

Java:
* (+) popular;
* (+) native for Hadoop;
* (-) too many lines of code. I remember the German language with very long words like "SchadschtoffFilteranlage" when reading a Java code. :-)

This choice was hard because no one knows Scala. But after a year and half I can say that Scala is a mix of Java and Python: a conciseness of Python and a power of Java.
