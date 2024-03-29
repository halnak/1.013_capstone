# Building on the previous attempt to size our battery, but eliminating the
# assumption that round trip efficiency is 100%.
library(tidyverse)

efficiency <- 0.875 # HN: I believe this is what Emily used?
#efficiency <- 0.5
discharge_efficiency <- efficiency
charge_efficiency <- efficiency
rt_efficiency <- efficiency * efficiency

# https://stackoverflow.com/questions/11693599/alternative-to-expand-grid-for-data-frames
expand.grid.df <- function(...) as.tibble(Reduce(function(...) merge(..., by=NULL), list(...)))

# Create a sine wave which approximates solar output of a 100MW nameplate
# capacity PV plant over a sunny day in July (basically the maximum it could be)
nameplate <- 808
#hour_begin <- 7
#solar_length <- 12 # hours 

placeholder <- rep(c(283.201,283.201,224.684,224.684,224.684,224.684,224.684,224.684,224.684,224.684,224.684,224.684,224.684,283.201,283.201,342.841,413.939,413.939,484.9,484.9,413.939,413.939,342.841,283.201),times=c(60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,61))
#placeholder
#seq(0, 1, by=1/10)
df <- tibble(
  time = seq(0, 24, by = 1/60),
  solar = placeholder
  #solar = nameplate * sin((time - hour_begin) * pi/solar_length)
)
df <- df %>%
  mutate(solar = ifelse(solar < 0, 0, solar))

# Base load: all 24 hours
# Peaking load: 5 pm to 9 pm, 4 hours
# Smoothed load: 5 am to 9 pm, 16 hours
# Current: base load
num_hours <- 24
lower_bound <- 5
upper_bound <- 21 # or 23?
power_caps <- tibble(power_cap = seq(0, nameplate, by = 0.25)) #not sure what this is?

df_middle <- expand.grid.df(df, power_caps) %>%
  group_by(power_cap) %>%
  mutate(plant_output = ifelse(time < lower_bound | time > upper_bound, 0, power_cap)) %>%
  mutate(difference = solar - plant_output) %>%
  mutate(energy_stored = ifelse(difference > 0, difference * charge_efficiency, difference / discharge_efficiency))

# Find the power capacity that best balances energy over the day
best <- df_middle %>%
  summarize(balance = abs(sum(energy_stored))) %>%
  filter(balance == min(balance))


# Find how much energy storage this requires
storage_req <- df_middle %>%
  filter(power_cap == best$power_cap) %>%
  filter(difference >= 0) %>%
  summarize(storage_mwh = sum(difference / 60) * charge_efficiency)#* charge_efficiency)

# Visualize balance
supplied_to_grid <- tibble(time = df$time) %>%
  mutate(output = ifelse(time < lower_bound | time > upper_bound, 0, best$power_cap))

intersections <- left_join(df, supplied_to_grid, by = "time") %>%
  filter(solar != 0) %>%
  mutate(diff = abs(solar - output)) %>%
  arrange(diff)
  

intersection_left <- min(intersections$time[1], intersections$time[2])
intersection_right <- max(intersections$time[1], intersections$time[2])
df_viz <- left_join(df, supplied_to_grid, by = "time") %>%
  mutate(rib_left_x = ifelse(time >= lower_bound & time <= intersection_left, time, NA)) %>%
  mutate(rib_right_x = ifelse(time <= upper_bound & time >= intersection_right, time, NA)) %>%
  mutate(rib_mid_x   = ifelse(time >= intersection_left & time <= intersection_right, time, NA))

top_fill <- "#8383FF"
bottom_fill    <- "#FF8783"
ggplot(df_viz, aes(x = time, y = solar)) +
  geom_line() +
  geom_line(aes(y = output)) +
  geom_ribbon(aes(x = rib_left_x, ymin = solar, ymax = output), fill = bottom_fill) +
  geom_ribbon(aes(x = rib_right_x, ymin = solar, ymax = output), fill = bottom_fill) +
  geom_ribbon(aes(x = rib_mid_x, ymin = output, ymax = solar), fill = top_fill) +
  scale_y_continuous(breaks = c(0, best$power_cap, nameplate)) +
  scale_x_continuous(breaks = seq(0, 24, by = 2)) +
  labs(x = "Hour", y = "Power (MW)") +
  theme_classic(base_size = 12.5)

#ggsave(filename = "../../thesis/draft/files/chap3/storage_principle.png", dpi = 900)
ggsave(filename = "test.png", dpi = 900)

storage_req %>% 
  mutate(hours_storage = storage_mwh * discharge_efficiency / power_cap)

