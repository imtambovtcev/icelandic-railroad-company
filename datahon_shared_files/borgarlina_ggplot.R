library(sf)
library(ggplot2)
library(tidyverse)

lina1 = sf::st_read("cityline_2025.geojson")

pop = read.csv("ibuafjoldi.csv")

pop$smasvaedi <- str_pad(pop$smasvaedi, width = 4, side = "left", pad = "0")

smallarea = sf::st_read("smasvaedi_2021.json")

dwellings = read.csv("ibudir.csv")

pop2024 <- pop |> filter(ar == 2024 & aldursflokkur == "10-14 ára" & kyn == 1)

all_dwellings = dwellings |> filter(framvinda == "Fullbúið") |> 
  group_by(smasvaedi) |> 
  summarise(Fjöldi = sum(Fjöldi))

pop2024_smallarea = smallarea |> left_join(pop2024, join_by("smsv" == "smasvaedi")) 

all_dwellings_smallarea = smallarea |> left_join(all_dwellings, join_by("fid" == "smasvaedi")) 

my_map = ggplot(pop2024_smallarea |> filter(nuts3 == "001")) + geom_sf(aes(fill = fjoldi, color = fjoldi)) + guides(fill = guide_none()) + geom_sf(data = lina1, fill = NA)

my_map

my_map = ggplot(all_dwellings_smallarea |> filter(nuts3 == "001")) + geom_sf(aes(fill = Fjöldi, color = Fjöldi)) + guides(fill = guide_none()) + geom_sf(data = lina1, fill = NA)

my_map
