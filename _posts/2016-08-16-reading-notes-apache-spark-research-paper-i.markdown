---
layout: article
title: "Apache Spark Research Paper I"
subtitle: "Spark: Cluster Computing with Working Sets"
tags: big-data spark reading-notes
---

For my summer internship at [Autodesk](http://www.autodesk.com), I have been heavily using Apache Spark for data analytics and machine learning. I believe a thorough understanding of the underlying principles and mechanisms of Apache Spark would be conducive to writing elegant and efficient Spark programs. Speaking of learning Spark, nothing is better than learning from the original authors. Therefore, I will read the three research papers out of The AMPLab at UC Berkeley that laid the groundwork for Apache Spark. I will start with the first paper [Spark: Cluster Computing with Working Sets](http://people.csail.mit.edu/matei/papers/2010/hotcloud_spark.pdf), published by Matei Zaharia in 2010.

<!--more-->

### Intro
- MapReduce pioneers the model of cluster computing that allows massively parallel computations, locality-aware scheduling (send the code to the data instead of the reverse) and fault tolerance
- However, Hadoop and MapReduce is lacking in two parts
    - **iterative jobs**: Hadoop needs to read/write disk for each iteration and incurs performance penalty
    - **interactive analytics**: each job runs as a separate MapReduce job and reads data, making is slow for interactive/repetitive queries

### Resilient Distributed Dataset (RDD) 
- read-only collections of objects partitioned across many machines
- RDD need not exist in storage, instead, a handle to the RDD contains information to compute the RDD is stored
- achieve fault tolerance through **lineage**: if a partition is lost, it can be rebuilt through the handle
- RDDs are lazy and ephemeral, they aren't materialized until an action is taken, and are discarded from memory after use
- RDD has two levels of persistence: 
    - **cache**: leaves the dataset lazy but will be cached in memory after the first action is taken
    - **save**: evaluate the action and writes to a file system

### Shared Variables
- Normally, when Spark runs a closure on a worker node, it copies the variables to the work as well, which increases the communication cost
- **Broadcast variables** 
    - used for large read-only piece of data
    - distribute the data to the workers once instead of packaging it with every closure
- **Accumulators** 
    - variables that workers can only "add" to using association operation, and only the driver can read
    - used for grouped data that is across many machines

### Performance 
- Logistic Regression and Alternating Least Squares, both of which are iterative algorithms, was ran on Spark and was order of magnitude faster than Hadoop, due to cached data.
