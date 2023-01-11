---
layout: article
title: "Apache Spark Research Paper II"
subtitle: "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing"
tags: big-data spark reading-notes
---

This follows up the last post and I will read the second Apache Spark paper [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](http://people.csail.mit.edu/matei/papers/2012/nsdi_spark.pdf), published by Matei Zaharia et al. in 2012.

<!--more-->

## Intro
- Cluster computing frameworks like MapReduce and Dryad are widely adopted since they provide the high level abstraction of parallel programming without worrying about work distribution and fault tolerance
- Frameworks like MapReduce lacks abstractions for leveraging distributed memory and makes them inefficient for iterative algorithms
- Resilient Distributed Datasets (RDDs) are fault tolerant, parallel data structures that let users explicitly persists intermediate results in-memory, controlling their partitioning to optimize data placement
- Existing abstractions for in-memory storage offer an interface based on **fine-grained** updates to mutable state. This makes the only way to provide fault tolerance being replicating the data across machines, which is too data intensive 
- RDDs provide an interface based on **coarse-grained** transformation (e.g., map, filter, join) that apply the same operation to many data items. this allows them to provide fault tolerance by logging the transformations used to build its **lineage** instead of replicating the actual data. If an RDD is lost, there's enough information about how it was derived from other RDDs so it can be recomputed
 
## RDDs

### Abstraction
- Read-only, partitioned collection
- Not materialized at all times
- Users can control **persistence** and **partitioning** of RDD

### Programming Interface
- Each dataset is represented as an object and interacted through an API
- Spark computes RDDs lazily the first time they are used in an action, so that it can pipeline transformations

### Advantage of the RDD Model
- Compare with distributed shared memory (DSM), where applications read and write to arbitrary locations (fine-grained transformation)in a global address space
- Main difference is that RDDs can only be created through coarse-grained transformation, which restricts RDDs to perform bulk writes, but allows for more efficient fault tolerance
- Immutable natures of RDDs let the system mitigate slow nodes by running backup copies of slow tasks just like MapReduce
- In bulk operations on RDDs, runtime can schedule task based on data locality, a.k.a "move processing to data" 
- RDDs degrade gracefully when there's not enough memory: partitions that do not fit in RAM are spilled on disk
 
### Applications not suitable for RDD
- Not suitable for applications that make asynchronous fine-grained updates to shared state, such as a storage system for a web application or an incremental web crawler
- Spark aims to provide an efficient programming model for batch analytics and is not suitable asynchronous applications

## Spark Programming Interface
- Developers write a driver program that connects to a cluster of workers
- The workers are long-lived processes that can store RDD partitions in RAM across operations
- Driver also keep tracks the RDDs' lineage
- Scala represents closures as Java objects, serialized and loaded them on another node to pass the closure across the network

## Representing RDDs
- A simple graph-based representation for RDDs that facilitates the goal of tracking lineage across a wide range of transformation
- Represent each RDD through a five pieces of information
    - a set of **partitions**: atomic pieces of the dataset
    - a set of **dependencies** on parent RDDs
    - a **function** for computing the dataset based on its parents
    - **metadata** about its partitioning scheme and data placement
- distinction between _narrow_ dependencies, where each partition of the parent RDD is used by at most one partition of the child RDD (join) and _wide_ dependencies, where multiple child partitions may depend on it (map)
    - narrow dependencies allows for pipelined execution on one node, while wide dependencies require shuffling of the partitions across the node
    - recovery after a node failure is more efficient with a narrow dependency

## Implementation
### Job scheduling
- Whenever a user runs an action on RDD, the scheduler examines that RDD's lineage to build a DAG of stages to execute, each stage consists of pipelined transformation with narrow dependencies. Boundaries of the stages are shuffling operations required for wide dependencies.
- Scheduler assigns tasks to machines based on data locality using **delay scheduling**: send task to node if a partition is available in memory on that node, otherwise, send to preferred locations
- Materialize intermediate records on the nodes holding parent partitions for wide dependencies

### Memory management 
- three levels of persistence:
    - in memory storage as **deserialized Java objects**: fastest performance
    - in memory storage as **serialized Java objects**: more memory-efficient representation at the cost of lower performance
    - on-disk storage
    - refer to [Understanding Spark Caching](http://sujee.net/2015/01/22/understanding-spark-caching/#.V76MRpMrJTY "Understanding Spark Caching") for reference
- For limited memory available, when a new RDD partition is computed but there's not enough space 
    - evict a partition from the least recently accessed RDD, unless this is the same RDD as the one with the new partition
    - If it's the same RDD, we keep the old partition in memory since it's likely that the partition already in memory will be needed soon

### Checkpointing
- long linage with wide dependencies are costly to recover, thus checkpointing maybe worthwhile
- Spark provides an API for checkpointing, but leaves the decision of which data to checkpoint to the user

