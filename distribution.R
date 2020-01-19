
library(tidyverse)

df <- read.csv("afbrs_transdf.csv") %>% 
             mutate(choice=as.character(choice))

#Data Distribution
ggplot(df, aes(region)) + 
    geom_bar(aes(fill = choice), width = 0.5) + 
    theme(axis.text.x = element_text(angle = 65, vjust = 0.6)) + 
    labs(x = "Region", y = "Count") + facet_grid(. ~ sector) + 
    coord_flip() + ggpubr::rotate_x_text()

ggplot(df, aes(region)) + 
    geom_bar(aes(fill = sector), width = 0.5) + 
    theme(axis.text.x = element_text(angle = 65, vjust = 0.6)) + 
    labs(x = "Region", y = "Count") +
    facet_grid(. ~ expectation) + coord_flip() + ggpubr::rotate_x_text()
