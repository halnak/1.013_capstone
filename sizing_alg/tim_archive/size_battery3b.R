library(tidyverse)
library(feather)
library(ggthemes)
library(cowplot)
library(extrafont)

results_raw  <- read_feather("size_battery_results.feather")
results <- results_raw %>%
  select(name, power_mw = power_cap_mw, energy_mwh = energy_storage_mwh, hours = hours_storage, balance = balance_absolute) %>%
  mutate_if(is.numeric, round, digits = 2)

profiles_raw <- read_feather("size_battery_profiles.feather")
solar <- profiles_raw %>%
  select(time, solar)
profiles <- profiles_raw %>% 
  gather(key = "name", value = "value", -time, -solar) %>%
  filter(name %in% c("c24_0_24", "c16_5_21", "c4_17_21"))
  #filter(name %in% c("c24_0_24", "c16_5_21", "c10_8_18", "c4_11_15", "c4_16_22", "c4_17_21"))

nice_names_fun <- function(name) {
  result <- name
  if (name == "c24_0_24") {
    result <- "Baseload Plant"
  } else if (name == "c4_17_21") {
    result <- "Peaker Plant"
  } else if (name == "c16_5_21") {
    result <- "Smoothed Output"
  }
  result
}

profiles <- profiles %>%
  mutate(name = map_chr(name, nice_names_fun))

results <- results %>% 
  arrange(name) %>%
  mutate(e_value = 3, e_breaks = 120, p_value = 3, p_breaks = 80) %>%
  filter(name %in% c("c24_0_24", "c16_5_21", "c4_17_21")) %>%
  #filter(name %in% c("c24_0_24", "c16_5_21", "c10_8_18", "c4_11_15", "c4_16_22", "c4_17_21"))
  mutate(name = map_chr(name, nice_names_fun))

profiles_plot <- ggplot(profiles, aes(x = time, y = value)) + 
  geom_line() +
  geom_line(aes(y = solar)) +
  facet_wrap(~name) +
  geom_text(data = results, 
            aes(x = e_value, 
                y = e_breaks, 
                label = paste("E:", round(energy_mwh, -1)))) +
  geom_text(data = results, 
            aes(x = p_value, 
                y = p_breaks, 
                label = paste("P:", round(power_mw)))) +
  labs(x = "Hour", 
       y = "Power (unit-invariant)",
       title = "Results of Battery Sizing Algorithm")

ggsave(profiles_plot, filename = "profiles_visual_85percent_eff_both_ways.png", dpi = 900)

#ggsave(filename = "../../thesis/draft/files/chap3/sizing_plot_results.png", dpi = 900)

# VISUALIZE PRINCIPLE OF BATTERY STORAGE FOR TWO CASES
const <- profiles %>%
  filter(name == "c24_0_24")


const_intersections <- const %>%
  filter(solar != 0) %>%
  mutate(diff = abs(solar - value)) %>%
  arrange(diff)

intersection_left <- min(const_intersections$time[1], const_intersections$time[2])
intersection_right <- max(const_intersections$time[1], const_intersections$time[2])

lower_bound <- 0
upper_bound <- 24

const_viz <- const %>%
  mutate(rib_left_x = ifelse(time >= lower_bound & time <= intersection_left, time, NA)) %>%
  mutate(rib_right_x = ifelse(time <= upper_bound & time >= intersection_right, time, NA)) %>%
  mutate(rib_mid_x   = ifelse(time >= intersection_left & time <= intersection_right, time, NA))

theme_set(theme_cowplot(font_size = 28, font_family = "Calibri"))

top_fill <- "#8383FF"
bottom_fill <- "#FF8783"
const_viz_plot <- ggplot(const_viz, aes(x = time, y = solar)) +
  geom_line() +
  geom_line(aes(y = value)) +
  geom_ribbon(aes(x = rib_left_x, ymin = solar, ymax = value), fill = bottom_fill) +
  geom_ribbon(aes(x = rib_right_x, ymin = solar, ymax = value), fill = bottom_fill) +
  geom_ribbon(aes(x = rib_mid_x, ymin = value, ymax = solar), fill = top_fill) +
  #scale_y_continuous(breaks = c(0, best$power_cap, nameplate)) +
  scale_x_continuous(breaks = seq(0, 24, by = 6)) +
  labs(x = "Hour", y = "Power (MW)", title = "Utilizing storage for a baseload plant")

const_viz_plot

# ggsave(plot = const_viz_plot, 
#        filename = "storage_illustration_const_power.png", 
#        dpi = 900,
#        width = 10.42,
#        height = 7.37)


peak <- profiles %>%
  filter(name == "c4_16_22")


peak_intersections <- peak %>%
  filter(solar != 0) %>%
  filter(time > 15.99) %>%
  mutate(diff = abs(solar - value)) %>%
  arrange(diff)

intersection <- min(peak_intersections$time[1], peak_intersections$time[2])

lower_bound <- 0
upper_bound <- 24

peak_viz <- peak %>%
  mutate(rib_left_x = ifelse(time >= lower_bound & time <= intersection, time, NA)) %>%
  mutate(rib_right_x = ifelse(time <= upper_bound & time >= intersection, time, NA)) %>%

top_fill <- "#8383FF"
bottom_fill <- "#FF8783"
peak_viz_plot <- ggplot(peak_viz, aes(x = time, y = solar)) +
  geom_line() +
  geom_line(aes(y = value)) +
  geom_ribbon(aes(x = rib_left_x, ymin = solar, ymax = value), fill = top_fill) +
  geom_ribbon(aes(x = rib_right_x, ymin = solar, ymax = value), fill = bottom_fill) +
  #scale_y_continuous(breaks = c(0, best$power_cap, nameplate)) +
  scale_x_continuous(breaks = seq(0, 24, by = 6)) +
  labs(x = "Hour", y = "Power (MW)", title = "Utilizing storage for a peaking plant")

peak_viz_plot

#ggsave(plot = peak_viz_plot, filename = "storage_illustration_peak_power.png", dpi = 900)

