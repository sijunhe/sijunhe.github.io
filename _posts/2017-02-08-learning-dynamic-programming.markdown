---
layout: article
title: "Learning Dynamic Programming"
tags: algorithms basics
---
Dynamic programming is a commonly-used method in a programmer's algorithm toolbox and one that I have yet to learn. Thus I am writing this post to document what I learned in Stanford's [Design and Analysis of Algorithms](https://www.coursera.org/learn/algorithm-design-analysis-2 "").

Dynamic programming is based on the idea of solving a complex problem by breaking it down into many simpler subproblems, solving each of the subproblems just once and storing the solution. When the same subproblem occurs again, the solution can be simply looked up from storage, thus saves time at the expense of space. 

<!--more-->

## Principles of Dynamic Programming  
1. Identify a small number of subproblems
2. Quickly and correctly solve "larger" subproblems, given the solutions to smaller subproblems
3. After solving all subproblems, the final solution can be quickly computed

## Example Problem 1: Weighted Independent Sets in Path Graphs

### Problem Statement
Path graphs are basically graphs like a linked list. The input for the Weighted Independent Sets (WIS) problem is a path graph with non-negative weights on the vertices. And the desired output would be the subset of non-adjacent vertices, also called independent set, with maximum total weight.

```python
o --- o --- o --- o --- 0
5     2     1     3     4
```

### Subproblem
Let $S$ be the max-weight independent set of the path graph $G$ and $V_n$ to be the last vertex of the path. Let $G'$ be the path graph $G$ with  $V_n$ deleted

- Case 1: If $V_n \notin S$, then $S$ is also the max-weight IS of $G'$.
- Case 2: If $V_n \in S$, then $S - $V_n$ is the max-weight IS of $G''$, which is $G$ with  $V_{n-1}$ and $V_n$ deleted

Hence, we know a max-weight IS $S$ must be either 

- a max-weight IS of $G'$ or
- a max-weight IS of $G''$ + $V_n$

### The Dynamic Programming Solution
Let $G_i$ be the first $i$ vertices of $G$ and we populate an array $A$ from left to right with $A[i]$ being the value of the max-weight IS of $G_i$. Let array $W$ store the weights of the vertices.

```python
A[0] = 0;
A[1] = W[1];
for i in range(2, n+1):
    A[i] = max(A[i-1], A[i-2] + W[i])    
```

### Complexity Analysis
Time complexity is $O(n)$ since it is constant time per iteration. Space complexity is also $O(n)$ from the memoization array $A$.

## Example Problem 2: The Knapsack Problem

### Problem Statement
The Knapsack problem mimics the situation where a thief breaks into a store with a knapsack. The knapsack can only carry certain weight so the thief needs to put in items with the most total value while stay under the knapsack's weight limit. 


Input: $n$ items, each with a non-negative integer value $v_i$ and a non-negative integer weight $w_i$. The weight constraint is $W$.

Output: a subset $S$ of all the items that maximizes $\sum_{i \in S} v_i$ subjected to $\sum_{i \in S} w_i \leq W$

### Subproblems
Let $S$ be the max-value solution to an knapsack with capacity $W$ and for item $n$ be the last of all the available items

- Case 1: If $n \notin S$, then S must also be the max-value solution to the first $(n-1)$ items
- Case 2: If $n \in S$, then then $S - \{n\}$ is the max-value solution to the first $(n-1)$ items and the knapsack with capacity of $W - w_n$

Hence, we know a max-value solution $V_{i,x}$ must be the maximum of 

- $V_{i-1,W}$, the max-value solution for the first $(n-1)$ items with the same capacity
- $V_{i-1,W-w_i} + v_i$, the sum of the max-value solution for the first $(n-1)$ items with the capacity of $W-w_i$ and value of the ith item $v_i$

### The Dynamic Programming Solution
Since we are searching all the possible combination of items and capacities, we use an 2D array $A$ for memoization. Let array $w$ and $v$ store the weights and values of the items.

```python
for x in range(0, W+1):
    A[0,x] = 0

for i in range(1, n+1):
    for x in range(0, W+1):
        A[i, x] = max(A[i - 1, x], A[i - 1, x - w[i] + v[i])    
```

### Complexity Analysis
Both the time complexity and the space complexity is also $O(nW)$, since there're $O(nW)$ subproblems and each can be solved in constant time and space.
