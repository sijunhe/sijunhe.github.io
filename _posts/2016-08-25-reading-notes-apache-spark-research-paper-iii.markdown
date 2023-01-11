---
layout: article
title: "Apache Spark Research Paper III"
subtitle: "Spark SQL: Relational Data Processing in Spark"
tags: big-data spark reading-notes
---

This follows up the last post and I will read the third Apache Spark paper [Spark SQL: Relational Data Processing in Spark](http://people.csail.mit.edu/matei/papers/2015/sigmod_spark_sql.pdf), published by Armbrust et al. in 2015 

<!--more-->

### 1. Intro
- Earliest big data processing systems like MapReduce give users a low-level **procedural** programming interface, which was onerous and required manual optimization by the user to achieve high performance
- New systems sought to offer relational interfaces to big data, like Pig, Hive, Dremel and Shark
- Spark SQL is bridges the gap between relational systems and procedural systems through two contributions: 
    - a **DataFrame** API that allows relational operations, similar to the data frame concept in R, but evaluates operations lazily
    - an optimizer **Catalyst**, that make it easy to dd data sources, optimization rules and data types for domains such as machine learning 

### 2. Programming Interface
Spark SQL runs as a library on top of Spark and exposed SQL interfaces through JDBC/ODBC, command-line console or DataFrame API
<img src="https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post6/spark_sql_diagram.png">

#### 2.1 DataFrame API
- A distributed collection of rows with the same schema (RDD of Row objects), equivalent to a table in relational database and can be manipulated with various relational operators
- DataFrames are lazy, in that each DataFrame object represents a **logical plan** to compute a dataset, but no execution occurs until the user calls an action

#### 2.2 Data Model
- Nested data model based on Hive and supports all major SQL data types
- Provides support for non-atomic data types like structs, arrays, maps and unions, which are not usually in RDMBS

#### 2.3 DataFrames vs SQL
- Easier to use due to integration in a full programming language
- Have full control building logical plan, which will be optimized by the Catalyst
- Able to store and access intermediate results

### 3. Catalyst Optimizer
- Catalyst’s extensible design had two purposes
    - Make it easy to add new optimization techniques and features to Spark SQL
    - Enable external developers to extend the optimizer
- Catalyst core contains a general library for representing trees and applying rules to manipulate them
- Several sets of rules to handle different phases of query execution: analysis, logical optimization, physical planning and code generation to compile queries to Java bytecode

#### 3.1 Trees
- Main data type in Catalyst, composed of nodes
- Each node has a node type and zero or more children

<img src="https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post6/spark_tree.png">

- This translate to 

```scala
Add(Attribute(x), Add(Literal(1), Literal(2))
```

#### 3.2 Rules
- Trees can be manipulated using rules, which are functions from a tree to another tree
- Mostly common approach is Scala’s pattern matching

```scala
tree.transform {
	case Add(Literal(c1), Literal(c2)) => Literal(c1+c2) 
	case Add(left, Literal(0)) => left
	case Add(Literal(0),right) => right
}
```

- Catalyst groups rules into batches, and executes each batch until the trees stops changing after applying its rules
- For example, for (x+0) + (3+3), a first batch might analyze an expression to assign types to all of the attributes, while a second batch might use these types to do constant folding


#### 3.3 Using Catalyst in Spark SQL
<img src="https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post6/spark_plan.png">


Catalyst’s general tree transformation framework contains four phases

- Analyzing a logical plan to resolve references
- Logical plan optimization
- Physical Planning, catalyst may generate multiple plans and compare them based on cost
- Code generation to compile parts of the query to Java bytecode

##### 3.3.1 Analysis
- Spark SQL begins with a relation to be computed, which may contain unresolved attributes references or relations
- Spark SQL uses Catalyst rules and a Catalog object that trakcs the tables in all data source to resolve the unresolved attributes
- Example query: SELECT col FROM sales
    - Looking up relations by name form the catalog, such as SELECT
    - Mapping named attributes, such as col, to the input provided 
    - Determining which attributes refer to the same value to give them a unique ID
    - Propagating and coercing types through expressions



##### 3.3.2 Physical Planning
- Spark SQL takes a logical plan and generates one or more physical plans, using physical operators that match the Spark execution engine
- Cost-based Optimization: selects a plan among all physical plans using a cost model
- Physical Planner also performs rule-based optimizations, such as pipelining projections or filters into one Spark map operation

##### 3.3.3 Code generation
- Need to support code generation to speed up execution, since Spark SQL operates on in-memory datasets where processing is CPU-bound
- Use Catalyst to transform a tree representing an expression in SQL to an AST for Scala code to evaluate that expression, and then compile and run the generated code
- Use a special feature of the Scala language: quasiquotes, which allows the programmatic construction of abstract syntax trees (AST)

##### 3.3.4 Data Sources
- Developers can define a new data soruce for Spark SQL, which expose varying degrees of possible optimization
- All data sources must implement a createRelation function that takes a set of key-value parameters and returns a BaseRelation object, which contains a schema and an optimal estimated size in bytes
- BaseRelation can implement one of several interfaces that let them expose varying degrees of sophistication
    - **TableScan**: return an RDD of Row objects for all the data in the table
    - **PrunedScan**: takes an array of column names to read and return Rows containing only those columns
    - **PrunedFilteredScan**: takes both desired column names and an array of Filter objects, which returns only rows passing each filters
    - **CatalystScan**: given a complete sequence of Catalyst expression trees to use in predicate pushdown

- The following data sources has been implemented:
    - **CSV files**, which simply scan the whole file but allow users to specify a schema
    - **Avro**, a self-describing binary format for nested data
    - **Parquet**, a columnar file format for which column pruning as well as filters are supported
    - **JDBC** data source, which scans ranges of a table from an RDBMS in parallel and pushes filters into the RDBMS to minimize communication

##### 3.3.5 User-Defined Types(UDTs)
- Allows user-defined types over relational tables
- Solve the issue by mapping UDTs to structures composed of Catalyst’s built-in types
- UDTs need to implement **datatype**, **serialize** and **deserialize** methods

### 4. Evaluation

#### 4.1 SQL Performance 
- Spark SQL is substantially faster than Shark and generally competitive with Impala

#### 4.2 DataFrames vs. Native Spark Code
- Spark SQL help non-SQL developers write simpler and more efficient Spark code through the DataFrame API, as Catalyst optimizes DataFrame operatons that are hard to do with hand written code
- Compare speed of computing the average of b for each value of a, for the dataset consists of 1 billion integer pairs (a,b) with 100,000 distinct values of a

```python
## Python API
sum_and_count = 
data.map(lambda x: (x.a, (x.b,1)))
       .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
       .collect()
## DataFrame API:
df.groupBy('a').avg('b')
```

- DataFrame API outperforms the hand written Python version by 12x, and outperforms Scala version 2x


#### 4.3 Pipeline Performance 
- DataFrame API improves performance in applications that combine relational and procedural processing by letting developers write all operations in a single program and pipelining computation across relational and procedural code
- Compare speed of two stage pipeline that selects a subset of text messages from a corpus and computes the most frequent words
- Separate SQL query followed by a Scala-based Spark job, similar to a Hive query followed by a Spark job, vs the DataFrame API
- DataFrame API pipelines improves performance 2x, since it avoids the cost of saving the whole result of the SQL query to an HDFS file as an intermediate dataset, because SparkSQL pipelines the **map** for the word count with the relational operators for the filtering


