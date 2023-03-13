# Using a simplified sinusoidal approach to size our battery.
library(tidyverse)

# https://stackoverflow.com/questions/11693599/alternative-to-expand-grid-for-data-frames
expand.grid.df <- function(...) as.tibble(Reduce(function(...) merge(..., by=NULL), list(...)))



# Create a sine wave which approximates solar output of a 100MW nameplate
# capacity PV plant over a sunny day in July (basically the maximum it could be)
nameplate <- 100
hour_begin <- 7
solar_length <- 12 # hours 

df <- tibble(
  time = seq(0, 24, by = 1/60),
  solar = nameplate * sin((time - hour_begin) * pi/solar_length)
)
df <- df %>%
  mutate(solar = ifelse(solar < 0, 0, solar))

# Get all permutations of the possible power capacities
power_caps <- tibble(power_cap = seq(0, nameplate, by = 0.25))
df_const_all_day <- expand.grid.df(df, power_caps) %>%
  group_by(power_cap) %>%
  mutate(difference = solar - power_cap)

# Find how each power capacity does in balancing energy
summed <- df_const_all_day %>%
  summarize(balance = abs(sum(difference)))

# Which power_capacity balances energy the best over the day?
best <- summed %>%
  filter(balance == min(balance))

## How much storage capacity does this require? ## 
storage_req <- df_const_all_day %>%
  filter(power_cap == best$power_cap) %>%
  filter(difference >= 0) %>%
  summarize(storage_mwh = sum(difference / 60))


# set up ribbons to define where we're taking the integral
ribbon_left <- df %>%
  filter(time < hour_begin)

ribbon_middle <- df %>%
  filter(time >= hour_begin, x < hour_begin + solar_length)

ribbon_right <- df %>%
  filter(time >= hour_begin + solar_length)

ggplot(df, aes(x = time, y = solar)) + 
  geom_line() + 
  geom_hline(yintercept = best$power_cap) +
  geom_ribbon(aes(x = time, ymin = 0, ymax = best$power_cap), data = ribbon_left, fill = "#BB000033") +
  theme_classic()

## How does this value change if we only supply power for the middle 14 hours? ##
num_hours <- 14

lower_bound <- hour_begin + (solar_length / 2) - (num_hours / 2)
upper_bound <- hour_begin + (solar_length / 2) + (num_hours / 2)
df_reduced_length <- expand.grid.df(df, power_caps) %>%
  group_by(power_cap) %>%
  mutate(plant_output = ifelse(time < lower_bound | time > upper_bound, 0, power_cap)) %>%
  mutate(difference = solar - plant_output)

# Find how each power capacity does in balancing energy
summed_reduced_length <- df_reduced_length %>%
  summarize(balance = abs(sum(difference)))

# Which power_capacity balances energy the best over the day?
best_reduced_length <- summed_reduced_length %>%
  filter(balance == min(balance))

ggplot(df, aes(x = time, y = solar)) + 
  geom_line() + 
  geom_hline(yintercept = best_reduced_length$power_cap) +
  theme_classic()

for14 <- 54.5
for12 <- 63.5

## How much storage capacity does this require? ## 
storage_req_reduced_length <- df_reduced_length %>%
  filter(power_cap == best_reduced_length$power_cap) %>%
  filter(difference >= 0) %>%
  summarize(storage_mwh = sum(difference / 60))
  
ggplot(df, aes(x = time, y = solar)) + 
  geom_line() + 
  geom_hline(yintercept = best_reduced_length$power_cap) + 
  theme_classic()
