---
layout: article
title: "San Francisco Crime Visualization"
subtitle: "Coursera Homework / Practicing visualization with R"
tags: r visualization kaggle
---
The purpose of this post is two-folded

- To complete the crime analytics visualization assignment from [Coursera: Communicating Data Science Results](https://www.coursera.org/learn/data-results)
- To try my hands at data visualization with ggplot2 and ggmap. 

<!--more-->

The dataset given by Coursera is actually a portion of the dataset from [Kaggle: San Francisco Crime Challenge](https://www.kaggle.com/c/sf-crime) so maybe I'll look into the Kaggle competition later as well. 

### Problem Statement
> In this assignment, you will analyze criminal incident data from Seattle or San Francisco to visualize patterns and, if desired, contrast and compare patterns across the two cities. You will produce a blog-post-style visual narrative consisting of a series of visualizations interspersed with sufficient descriptive text to make a convincing argument. You will use real crime data from Summer 2014 one or both of two US cities: Seattle and/or San Francisco.

### Visualization
For the Coursera assignment, I will produce a map of San Francisco with the top committed crimes and also look into the neighborhoods susceptible to particular kinds of crimes. 

```r
library(dplyr)
library(ggmap)
library(ggplot2)
data <- read.csv("../datasci_course_materials/assignment6/sanfrancisco_incidents_summer_2014.csv")
data <- subset(data, select = c(IncidntNum, Category, X, Y))
##getting ggmap basemap from Google Map
map <- get_map(location = c(lon = -122.4367, lat = 37.7733), zoom = 13, maptype = "roadmap", color = "bw")
``` 

After importing the dataset, we will take a subset of the data since only the crime categories and the crime coordinates are needed for this visualization. For the visualization, we will use a popular R package called **ggmap**, which is designed for spatial visualization in combination with **ggplot2**. The map returned from the get_map command looks like this: 

![png](	
https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post4/basemap.jpeg)

Then, we will start putting data points on the base map. However, before we do that, we need to take a look at the data first. 

```r
## use summarize command from dplyr package to group the crimes by their categories and do a number count
counts = summarize(group_by(data, Category), Counts=length(Category))
## descending order
counts = counts[order(-counts$Counts),]
top5 <- data[data$Category %in% counts$Category[1:5],]
```


	          Category Counts
	            (fctr)  (int)
	1    LARCENY/THEFT   9466
	2   OTHER OFFENSES   3567
	3     NON-CRIMINAL   3023
	4          ASSAULT   2882
	5    VEHICLE THEFT   1966
	6         WARRANTS   1782
	7    DRUG/NARCOTIC   1345
	8   SUSPICIOUS OCC   1300
	9   MISSING PERSON   1266
	10 SECONDARY CODES    442
	..             ...    ...

It appears that theft is the most common crime in San Francisco, followed by "other offenses" and "non-criminal" crimes (not sure what those mean). Assault and Drug-related crimes are also quite common. Let's put these on the map and see how they distribute spatially. 

```r
Top5Crime <- ggmap(map) + 
        geom_point(data = top5, aes(x=X, y=Y, colour=factor(Category)), alpha=0.10) + 
        ggtitle("Top 5 Crimes in San Francisco in Summer 2014") +
        theme_light(base_size=20) +
        guides(colour = guide_legend(override.aes = list(alpha=1.0, size=6.0), title="Type of Crime")) +
        scale_colour_brewer(type="qual",palette="Paired") +  
        theme(axis.line=element_blank(),
              axis.text.x=element_blank(),
              axis.text.y=element_blank(),
              axis.ticks=element_blank(),
              axis.title.x=element_blank(),
              axis.title.y=element_blank())
```

![png](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post4/top5.jpeg)

Due to the clustering of the data points, the coloring doesn't work as well as expected but we can still get a good sense of how the crimes distributed spatially. The notorious Tenderloin neighborhood has the most concentrated crimes and that extends to the Downtown area north of it. The number of crimes declines significantly after crossing the Market St. to the south and the US101 to the west. The other interesting feature worth mentioning is that crimes decrease as we go east or north from the Tenderloin area into the Nob Hill and Financial District neighborhoods, but increase as we approach the water front. 

Now let's take a particular look at Larceny/Theft, the most common crime in San Francisco. The spatial distribution of theft matches the top 5 crimes distribution quite well. Apart from the Tenderloin neighborhood, the areas with high theft counts are SOMA, Western Addition and Mission St. 
![png](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post4/larceny.jpeg)

Now we'll move away from the more common crimes and take a look at robbery, which is not among the top 10 most-committed crimes in San Francisco. While the count is still higher in Tenderloin neighborhood compared with other areas, robberies are actually pretty spread-out in the city. 

![png](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post4/robbery.jpeg)

For drug-related crimes, it is an entirely different story. The crimes related to drugs or narcotics are highly concentrated in the Tenderloin neighborhood north of Market St.and south of Geary St, while the distribution out of the Tenderloin area is quite sparse. I am guessing that drug-related crimes are usually organized by gangs, which tend to stay in certain area; robbery could be committed by any individual criminals, which makes it more spread out.   

![png](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post4/drug.jpeg)

So, the takeaway advice after reading the post is clearly to avoid the Tenderloin area when traveling to the city by the bar. 

### Citation
The R code I used was largely borrowed from Ben Hamner in his [script](https://www.kaggle.com/benhamner/sf-crime/san-francisco-top-crimes-map) of San Francisco Top Crimes Map on Kaggle. 
