---
layout: page
title: "Categories"
date: 2013-07-28 23:11
comments: true
sharing: false
footer: true
---
<ul>
{% assign sorted_categories = site.categories | sort %}
{% for item in sorted_categories %}
    <li><a href="/blog/categories/{{ item[0] }}/">{{ item[0] }}</a> [ {{ item[1].size }} ]</li>
{% endfor %}
</ul>
