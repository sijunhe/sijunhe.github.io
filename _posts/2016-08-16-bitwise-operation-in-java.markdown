---
layout: article
title: "Bitwise Operation in Java"
tags: java basics
---
Recently, I have been playing with [LeetCode](https://leetcode.com/ "") a bit in preparation for the upcoming job hunting season and I found a class of problems related to **bitwise operations**. While I may never use low level bitwise operations as a data scientist, they are quite fun to work with. Plus it would be virtually impossible to solve bitwise operation problems without reading into it before hand. So here are my reading notes for bitwise operations in Java.

<!--more-->

### Bit Operator

- #### Bitwise AND ( & ) 
The & operator applies the logical AND bit by bit. If both bits are 1, it returns 1. Otherwise, return 0.
```python
 0000 1100(12)
      &
 0000 1010(10)
      =
 0000 1000 (8)
```

- #### Bitwise OR ( | ) 
The | operator applies the logical inclusive OR bit by bit. If both bits are 0, it returns 0. Otherwise, return 1.
```python
 0000 1100(12)
      |
 0000 1010(10)
      =
 0000 1110(14)
```

- #### Bitwise XOR ( ^ )
The ^ operator applies the logical exclusive OR bit by bit. If returns 1 only if one of the bits is 1 and returns 0 otherwise.
```python
 0000 1100(12)
      ^
 0000 1010(10)
      =
 0000 0110 (6)
```

- #### Bitwise Complement ( ~ )
The ~ operator applies the logical NOT bit by bit. It "flips" each bit. The complement operator also follows the formula \(\sim x = - x - 1 \), if the negatives are stored as [Two's Complement](https://en.wikipedia.org/wiki/Two%27s_complement).
```python
 ~  0000 1100(12)   
        =
    1111 0011 = 0000 1100 + 1 = (-13)
```

### Bit Shift

- #### Left Shift ( << )
The << operator shifts the current bit pattern to the left and pads the empty bits on the right with 0. For each bit shifted to the left, it is equivalent to multiplying the original number by 2. To put into formula, \( x << y = x * 2^y\)
```python
  0000 1100(12) 
     << 1
  0001 1000(24)
```

- #### Right Shift ( >> )
The >> operator shifts the current bit pattern to the right and pads the empty bits on the left with the most significant bit, which is the sign bit. This allows the operator to preserve the sign of the number. For each bit shifted to the right, it is equivalent to divide the original number by 2. To put into formula, \( x >> y = \frac{x}{2^y}\)
```python
  0000 1100(12) 
     >> 1
  0000 0110(6)

  1111 0100(-12) 
     >> 1
  1111 1010(-6)
```

- #### Zero Fill Right Shift ( >>> )
The >>> operator behaves very similar to the >> operator. However, instead of padding with the most significant bit, it pads with 0. Hence, this operator is not sign-preserving. If the original number is positive, it is still equivalent to divide the original number by 2 for each bit shifted. But it doesn't work for negative numbers.
```python
  0000 1100(12) 
     >>> 1
  0000 0110(6)

  1111 0100(-12) 
     >>> 1
  0111 1010(122)
```