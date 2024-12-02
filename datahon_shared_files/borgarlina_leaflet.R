library(sf)
library(leaflet)

lina1 = sf::st_read("cityline_2025.geojson")

pop = read.csv("ibuafjoldi.csv")

pop$smasvaedi <- str_pad(pop$smasvaedi, width = 4, side = "left", pad = "0")

smallarea = sf::st_read("smasvaedi_2021.json")

dwellings = read.csv("ibudir.csv")

smallarea_wgs84 <- st_transform(smallarea, 4326)
lina1_wgs84 <- st_transform(lina1, 4326)

pop2024 <- pop |> filter(ar == 2024 & aldursflokkur == "10-14 ára" & kyn == 1)

all_dwellings = dwellings |> filter(framvinda == "Fullbúið") |> 
  group_by(smasvaedi) |> 
  summarise(Fjöldi = sum(Fjöldi))

pop2024_smallarea = smallarea_wgs84 |> left_join(pop2024, join_by("smsv" == "smasvaedi")) 

all_dwellings_smallarea = smallarea_wgs84 |> left_join(all_dwellings, join_by("fid" == "smasvaedi")) 

pal <- colorNumeric("viridis", NULL)

leaflet(all_dwellings_smallarea |> filter(nuts3 == "001")) |> 
  addTiles() |> 
  addPolygons(stroke = FALSE, smoothFactor = 0.3, fillOpacity = 0.7,
              fillColor = ~pal(Fjöldi),
              label = ~paste0(smsv_label, ": ", formatC(Fjöldi, big.mark = ","))) |> 
  addMarkers(data=lina1_wgs84)
