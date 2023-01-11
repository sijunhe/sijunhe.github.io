---
layout: article
title: "Quora Challenges - Upvotes"
tags: python algorithms
---

Quora Challenges are basically a set of programming problems posted on the [website](https://www.quora.com/challenges), used as like a backdoor of recruiting. [Rumor](https://www.quora.com/Is-it-necessary-to-solve-a-programming-challenge-to-get-an-internship-interview-with-Quora) has it that whoever can solve one of these would guarantee an interview with Quora, skipping the resume selection process.   

<!--more-->

## Problem Statement
> At Quora, we have aggregate graphs that track the number of upvotes we get each day. As we looked at patterns across windows of certain sizes, we thought about ways to track trends such as non-decreasing and non-increasing subranges as efficiently as possible.     

> For this problem, you are given N days of upvote count data, and a fixed window size K. For each window of K days, from left to right, find the number of non-decreasing subranges within the window minus the number of non-increasing subranges within the window.  

> A window of days is defined as contiguous range of days. Thus, there are exactly N−K+1
 windows where this metric needs to be computed. A non-decreasing subrange is defined as a contiguous range of indices [a, b], a < b, where each element is at least as large as the previous element. A non-increasing subrange is similarly defined, except each element is at least as large as the next. There are up to K(K − 1) / 2 of these respective subranges within a window, so the metric is bounded by [−K(K−1) / 2, K(K − 1) / 2].
>>[Quora Challenges](https://www.quora.com/challenges#upvotes)   



## Approach 
- The problem is asking for the number of subranges, not the longest subranges. For example, [1, 2, 3] has 3 non-decreasing ranges [1, 2], [2, 3], [1, 3]. Therefore the common approach of iterating a pointer through the window while counting subrange length until comparison condition breaks is not valid. Instead, we need to generate a list of all the subranges in a window and count the combination of them. For example, a 10 day non-decreasing subrange has 9 + 8+ 7+ 6 + 5 + 4 + 3 + 2 + 1 = 45 non-decreasing subranges.   

- We could do a Brute Force algorithm of computing the number of subranges separately for each window and loop the algorithm through every window. But the algorithm for the subranges would be O(K) within a window since it has to touch each element in the window at least once. The entire algorithm would then be O(K(N - K + 1)) = O(KN), which is way to slow.   

- Instead, we can compute the number of subranges for the first window and update those subranges. The update is very efficient O(1) since we are just removing an element in the back and adding an element in the front and we just need to do a constant amount of comparisons to update the front and back subranges.    


## Code 
```python
# This file solves the upvotes challenge on Quora Engineering Challenges
# https://www.quora.com/challenges#upvotes

# Find all Non-decreasing range in a window and append them into a list
# parameters: start, end: starting and ending point of the window
# return: a list containing all non-decreasing subranges in the window, subranges are in the form of [first, last]
def findAllNonDecr(start, end):
    output = []
    # first = starting point of an subrange
    # last = ending pointer of an subrange
    first = last = start

    for num in range(start + 1, end ):
        if list[num] >= list[last]:
            last = num
        else:
            # append the subrange to the subrange list if the length is at least 2 days
            if first != last:
                output.append([first, last])
            # reset first and last to the pointer
            first = last = num

    # if subrange continues through the last element
    if first != last:
        output.append([first, last])

    return output

# Same function as last one, but find all non-increasing range
def findAllNonIncr(start, end):
    output = []
    first = last = start

    for num in range(start + 1, end):
        if list[num] <= list[last]:
            last = num
        else:

            if first != last:
                output.append([first, last])

            first = last = num

    # if subrange continues through the last element
    if first != last:
        output.append([first, last])

    return output

# calculate the number of subranges within the non-Decr/Incr range, given the start and the end of the subrange
def sumSubrange(output):
    if (len(output) ==0): return 0
    sum = 0
    # find the number of subranges for each subrange in the list
    for pair in output:
        diff = pair[1] - pair[0] + 1
        # The number of subrange in a N-length subrange is 1 + 2 + ... + N - 1 = (N - 1) * N/2
        sum += (diff - 1) * diff / 2

    return sum


# move the window down the list, updating the non-decreasing range list as well as the number of non-decreasing subranges
def moveNonDecr(start, end, output, sum):

    # If there weren't any subranges from last window, do a brand new search in current window
    if len(output) == 0:
        newOutput = findAllNonDecr(start + 1, end + 1)
        newSum = sumSubrange(newOutput)
        return newOutput, newSum

    firstSubrange = output[0] # first subrange of previous window
    lastSubrange = output[-1] # last subrange of previous window

    # take care of boundary condition in the back
    if firstSubrange[0] == start:

        # if the start of the window was the start of the first subrange, then the subrange will shift by one
        sum -= (output[0][1] - output[0][0])
        output[0][0] += 1

    # # take care of boundary condition in the front
    # if the end of the window was the end of the last subrange
    if lastSubrange[1] == end - 1:

        # if the end of the subrange continues, shift the subrange by one
        if list[end] >= list[end-1]:
            sum += (output[-1][1] - output[-1][0] + 1)
            output[-1][1] += 1

    # if the end of the window was not the end of the last subrange, see if the last element from last window and the new element would form a new subrange
    else:
        if list[end] >= list[end -1]:

            output.append([end-1, end])
            sum += 1

    # if boundary condition in the back makes the first subrange of only length 1, delete that subrange from list

    if output[0][0] == output[0][1]:
        output.pop(0)

    return output, sum

# Same function as last one, but update all non-increasing range
def moveNonIncr(start, end, output, sum):

    if len(output) == 0:
        newOutput = findAllNonIncr(start + 1, end + 1)
        newSum = sumSubrange(newOutput)
        return newOutput, newSum

    firstSubrange = output[0] # first subrange of previous window
    lastSubrange = output[-1] # last subrange of previous window

    # take care of left edge condition
    if firstSubrange[0] == start:

        # if the start of the window was the start of the first subrange, then the subrange will shift by one
        sum -= (output[0][1] - output[0][0])
        output[0][0] += 1

    # take care of right edge condition
    # if the end of the window was the end of the last subrange
    if lastSubrange[1] == end - 1:

        # if the end of the subrange continues, shift the subrange by one
        if list[end] <= list[end-1]:
            sum += (output[-1][1] - output[-1][0] + 1)
            output[-1][1] += 1

    # if the end of the window was not the end of the last subrange, try if the last two element would form a new subrange
    else:
        if list[end] <= list[end -1]:

            output.append([end-1, end])
            sum += 1

    # if that makes the subrange only 1 number, delete that subrange from list
    if output[0][0] == output[0][1]:
        output.pop(0)

    return output, sum

# input for N and K as int
n, k = raw_input().split()
n, k = int(n), int(k)

# input for N positive Integers of upvote counts as int list
global list
list = raw_input().split()
list = [int(i) for i in list]

# initialize a pointer for the start and end of the size K window
start = 0
end = 0 + k
outputNonDecr = findAllNonDecr(start, end)
outputNonIncr = findAllNonIncr(start, end)
sumNonDecr = sumSubrange(outputNonDecr)
sumNonIncr = sumSubrange(outputNonIncr)
print sumNonDecr - sumNonIncr

while end != n:
    outputNonDecr, sumNonDecr = moveNonDecr(start, end, outputNonDecr, sumNonDecr)
    outputNonIncr, sumNonIncr = moveNonIncr(start, end, outputNonIncr, sumNonIncr)
    start += 1
    end += 1
    print sumNonDecr - sumNonIncr
```

All 29 test cases passed! 

## Potential Areas of Improvement 
- Maybe a window can be scanned once for both non-decreasing subrange and non-increasing subrange by keeping track of more pointers. I am not certain, without giving much thought.   

- Maybe the updating of both non-decreasing subrange and non-increasing subrange can be done simultaneously. 