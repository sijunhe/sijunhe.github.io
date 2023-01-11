---
layout: article
title: "The Stack and The Heap"
tags: miscellaneous basics
---

As a self-taught programmer, my biggest weakness has always been in the computer systems area, since I never had the luxury of taking a series of foundational systems courses like the [CS107](https://web.stanford.edu/class/cs107/ "CS107") - [CS110](http://web.stanford.edu/class/cs110/ "CS110") - [CS140](http://cs140.stanford.edu/ "CS140") series at Stanford. While I don't find them indispensable in pursuing my interest of data science yet, it is always to catch up on things that I don't know. So here are some reading notes for the comparison between the Stack and the Heap in RAM. 

<!--more-->

### The Stack 
- A region of your computer's memory that stores temporary variables created by each function
- Freeing memory is easy since it's LIFO, just need to adjust one pointer
- Very fast and efficient access since the Stack is managed by the CPU
- The size is limited
- When a function exits, **all** variables defined pushed on the Stack by that function is freed 
- Variables in the Stack can only be accessed locally, unless specified by the keyword **static** in C

### The Heap 
- A region of your computer's memory that is not managed automatically
- Slower access since it's not as tightly managed by the CPU as the Stack 
- The size is unlimited
- Need to manually allocate and free memory, like **malloc()** and **free()** in C. Failing to do so will result in memory leak 
- Variables available globally

