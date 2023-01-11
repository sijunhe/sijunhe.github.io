---
layout: article
title: "Friends You Might Know"
subtitle: "Friend Recommendation with Hadoop"
tags: java hadoop big-data algorithms
---
This is a problem from [CS246](http://web.stanford.edu/class/cs246/): Mining Massive Datasets from Stanford University. The idea behind the original "People You Might Know" friendship recommendation algorithm is that if two people share a high number of mutual friends, they are more likely to know each other and thus the system would recommend that they connect with each other. 

<!--more-->


## Problem Statement

The input here can be found [here](https://s3-us-west-1.amazonaws.com/sijunhe-blog/data/soc-LiveJournal1Adj.txt), which is in the form of [User][Tab][Friends]", where [Friends] is a comma-separated list of Users. The output will be in terms of [User][Tab][Recommendations], where [Recommendations] is a comma-separated list of users that are recommended friend.   

We could use a algorithm that for each user A, it recommends N = 10 users who are not already friends with A, but share the largest number of mutual friends in common with A.  

## Approach and Pseudocode

The key is that for each user, the algorithm needs to find out the users who share mutual friends with him. Then, the algorithm needs to sort the users by the number of mutual friends in descending order and to make sure not to recommend the users who are already friends with A. Clearly, this is a highly parallelizable problem since for each user the computation is entirely separated. This would make a great MapReduce problem. 

####Terminology
For each user A:  

Degree 1 friends: the users who A is already friends with  
Degree 2 friends: the users who shares a mutual friend with A.  
Degree 2 friends could potentially be degree 1 friends as well and we are interested in the users who are degree 2 friends with A, but not degree 1 friends. 

### Approach
Since we are using MapReduce, we want the reducer to have the key as A, and the value as an [Iterable](https://docs.oracle.com/javase/7/docs/api/java/lang/Iterable.html) of degree 1 friends and degree 2 friends.  
 
Example input: A B,C,D  

Mapper:  

- Computing degree 1 friends for A would be straightforward, as they would just be the users in the comma-separated friends list. The mapper would just emit the key values pairs of (A, B), (A, C) and (A, D)  
- Computing degree 2 friends is a little trickier. Here we will use A as the key, but as the mutual friend. For example, from A's friends list, we know B and C both have A as a mutual friend. Therefore B is a degree 2 friend of C and vice versa. The mapper would emit (B, C), (C, B), (B, D), (D, B), (C, D) and (D, C) 

Reducer:   
 
- The reducer receives the key, which is an user, and an Iterable of degree 1 and degree 2 friends of the key    
- The reducer needs to count the number of times that a degree 2 friend pair shows up is the number of their mutual friends
- The reducer then needs to sort the degree two friends of the key by the number of mutual friends
- The reducer needs to make sure that degree 1 don't get recommended

### Pseudocode 

```
Map: 
	
	for each user, friendList:
		for each friend i in friendList:
			emit (user, (i, 1)) 
			for each friend j in friendlist: 
			emit (j, (i, 2))
			emit (i, (j, 2))

Reduce: 
	
	for each user, friendDegreeList:

		for each (friend, degree) in  friendDegreeList:
			if degree == 1: hashmap.add(friend, -1)
			else:
				if (friend in hashmap):
					hashmap[friend] += 1
				else:
					hashmap.add(friend, 1)

		for each entry in hashmap:
			if (value != -1): add to list

		sort list
                output top N

```


## Hadoop MapReduce Program  
```java
package edu.stanford.cs246.friendrecommendation;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.PriorityQueue;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class FriendRecommendation extends Configured implements Tool {
   public static void main(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      int res = ToolRunner.run(new Configuration(), new FriendRecommendation(), args);
      System.exit(res);
   }

   @Override
   public int run(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      Job job = new Job(getConf(), "FriendRecommendation");
      job.setJarByClass(FriendRecommendation.class);
      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(PairWritable.class);

      job.setMapperClass(Map.class);
      job.setReducerClass(Reduce.class);

      job.setInputFormatClass(KeyValueTextInputFormat.class);
      job.setOutputFormatClass(TextOutputFormat.class);

      FileInputFormat.addInputPath(job, new Path(args[0]));
      FileOutputFormat.setOutputPath(job, new Path(args[1]));

      job.waitForCompletion(true);
      
      return 0;
   }
   
   public static class Map extends Mapper<Text, Text, Text, PairWritable> {

      @Override
      public void map(Text key, Text value, Context context) throws IOException, InterruptedException {;
         if (value.getLength() != 0) {
        	 String[] friendsList = value.toString().split(",");
        	 for (int i = 0; i < friendsList.length; i ++) {
        		 //adding all degree 1 friends to keep track of the people who are already friends and don't need to be recommended
        		 context.write(key, new PairWritable(new Text(friendsList[i]), new IntWritable(1)));
        		 for (int j = i + 1; j < friendsList.length; j++) {
        			 //adding all potential degree 2, which are people in the list who are both friends with the person (key)
        			 context.write(new Text(friendsList[i]), new PairWritable(new Text(friendsList[j]), new IntWritable(2)));
        			 context.write(new Text(friendsList[j]), new PairWritable(new Text(friendsList[i]), new IntWritable(2)));
        		 }
        	 }
         }	 
      }
   }

   public static class Reduce extends Reducer<Text, PairWritable, Text, Text> {
      @Override
      public void reduce(Text key, Iterable<PairWritable> values, Context context) throws IOException, InterruptedException {
    	  Iterator<PairWritable> itr = values.iterator();
    	  //use a hashmap to keep track of the friends of the person and the count
    	  HashMap<String, Integer> hash = new HashMap<String, Integer>(); 
          while (itr.hasNext()) {
        	  PairWritable curEntry = itr.next();
        	  String friendName = curEntry.getFriend();
        	  // if this is a degree 1 friend, identify it with -1 and later delete it
        	  if (curEntry.getDegree() == 1) hash.put(friendName, -1);
        	  else {
        		  if (hash.containsKey(friendName)) {
        			  //if person already in the list
        			  if (hash.get(friendName) != -1) {
        				  //make sure that the friends pair are not degree 1 friends
        				  // if not, increase the count by 1
        				  hash.put(friendName, hash.get(friendName) + 1);
        			  }
        		  }
        		  else hash.put(friendName, 1);
        	  }
          }
          
          // remove all degree 1 friend, sort top 10 with a top 10 heap (implemented by PriorityQueue), output 
          PriorityQueue<Entry<String, Integer>> top10heap = new PriorityQueue<Entry<String, Integer>>(10, new Comparator<Entry<String, Integer>>() {

			@Override
			public int compare(Entry<String, Integer> o1,
					Entry<String, Integer> o2) {
				return o2.getValue() - o1.getValue();
			}
          }); 
          
          for (Entry<String, Integer> pairs: hash.entrySet()) {
        	  if (!pairs.getValue().equals(-1)) top10heap.add(pairs);
          }
          StringBuffer output = new StringBuffer();
          int count = 0;
          int size = top10heap.size();
          while (!top10heap.isEmpty()) {
        	  output.append(top10heap.poll().getKey());
        	  if (count >= 9 || count >= size-1) break;
        	  count ++;
        	  output.append(",");
          }
          context.write(key, new Text(output.toString()));
      }
   }
   
   
   /*
    * the implementation of a friend and the degree with the WritableComparable interface required as a value 
    */
   public static class PairWritable implements Writable {
	   private Text friend;
	   private IntWritable degree;
	   
	   public PairWritable() {
		   this.friend = new Text();
		   this.degree = new IntWritable();
	   }
	   
	   public PairWritable(Text friend1, IntWritable degree) {
		   this.friend = friend1;
		   this.degree = degree;
	   }

	   @Override
	   public void readFields(DataInput in) throws IOException {
		   this.friend.readFields(in);
		   this.degree.readFields(in);
	   }

	   @Override
	   public void write(DataOutput out) throws IOException {
		   friend.write(out);
		   degree.write(out);
	   }
	   
	   public int getDegree() {
		   return degree.get();
	   }
	   
	   public String getFriend() {
		   return friend.toString();
	   }
   }

}


```
