import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import feather

efficiency = 0.85
discharge_efficiency = efficiency
charge_efficiency = efficiency
rt_efficiency = efficiency * efficiency

# Create a sine wave which approximates solar output of a 100MW nameplate
# capacity PV plant over a sunny day in July (basically the maximum it could be)
nameplate = 100
solar_hour_begin = 7
solar_length = 12 # hours

df = pd.DataFrame({'time' : np.linspace(0, 24, num = 24*60+1)})

df = df.assign(solar = lambda x: nameplate * np.sin((x.time - solar_hour_begin) * np.pi/solar_length))
df['solar'] = np.where(df['solar'] >= 0, df['solar'], 0)

plt.plot(df['time'], df['solar'])
plt.show()

# generate different output profiles, as your battery sizing will depend
# on the profile
profile_names = list()

# off by one error?
for i in np.linspace(24, 4, num = (24-4)//2 + 1):
    mid_solar_day = solar_hour_begin + solar_length/2
    if i == 24:
        hour_begin = 0
    else:
        hour_begin = mid_solar_day - i/2

    length = i
    profile = []
    for j in np.linspace(0, 24, num = 24*60+1):
        if j < hour_begin:
            profile.append(0)
        elif j < (hour_begin + length):
            profile.append(1)
        else:
            profile.append(0)

    rep_string = "c" + str(int(i)) + "_" + str(int(hour_begin)) + "_" + str(int(hour_begin + length))
    df[rep_string] = profile
    profile_names.append(rep_string)

# add two peaking plants (6 and 4 hours, in the evening)
profile = []
hour_begin = 16
length = 6
for j in np.linspace(0, 24, num = 24*60+1):
    if j < hour_begin:
        profile.append(0)
    elif j < hour_begin + length:
        profile.append(1)
    else:
        profile.append(0)
rep_string = "c" + str(int(length)) + "_" + str(int(hour_begin)) + "_" + str(int(hour_begin + length))
df[rep_string] = profile
profile_names.append(rep_string)

profile = []
hour_begin = 17
length = 4
for j in np.linspace(0, 24, num = 24*60+1):
    if j < hour_begin:
        profile.append(0)
    elif j < hour_begin + length:
        profile.append(1)
    else:
        profile.append(0)
rep_string = "c" + str(int(length)) + "_" + str(int(hour_begin)) + "_" + str(int(hour_begin + length))
df[rep_string] = profile
profile_names.append(rep_string)
#df
#profile_names
results = pd.DataFrame({'name' : profile_names,
                        'power_cap_mw' : np.zeros(len(profile_names)),
                        'energy_storage_mwh' : np.zeros(len(profile_names)),
                        'hours_storage' : np.zeros(len(profile_names)),
                        'balance_absolute' : np.zeros(len(profile_names))})

# profile_names
for profile_name in profile_names:
#for profile_name in ['c24_0_24']:
    # evaluate each potential battery size (rated by power) to see which one
    # balances energy best
    best_balance = np.Inf
    best_power_cap = 0
    best_profile = []
    best_difference = []
    plt.plot(df['time'], df[profile_name])
    plt.show()
    for power_cap in np.linspace(0, 200, num = 200*4):

        # for your output profile, always use the battery to full capacity
        # that way, you don't have any wasted capacity.
        profile = np.where(np.isclose(df[profile_name], 1), power_cap, 0)

        # use whatever solar you have available to meet the current demand.
        # when you have excess solar, charge your battery. when not enough,
        # discharge the battery.
        difference = df['solar'] - profile # positive when you need to charge

        # charge or discharge your battery depending on whether or not you
        # have excess or not enough.
        energy_stored = np.where(difference > 0,
                                 difference * charge_efficiency,
                                 difference / discharge_efficiency)

        # looking to minimize the difference between stored and discharged.
        # because energy_stored is positive and negative, summing it
        # gives us this number.
        balance = np.abs(np.sum(energy_stored))

        if balance < best_balance:
            best_balance = balance
            best_power_cap = power_cap
            best_profile = profile
            best_difference = difference

    # https://stackoverflow.com/questions/43193880/how-to-get-row-number-in-dataframe-in-pandas

    df[profile_name] = best_profile
    row_num = results.loc[results['name'] == profile_name].index[0]
    results.set_value(row_num, 'power_cap_mw', best_power_cap)

    # divide by 60 because our profiles are by the minute, not by the hour
    results.set_value(row_num,
                      'energy_storage_mwh',
                      np.sum(np.where(best_difference > 0, best_difference,
                                      0) * charge_efficiency) / 60)
    results.set_value(row_num, 'hours_storage', results.get_value(row_num, 'energy_storage_mwh') * discharge_efficiency / results.get_value(row_num, 'power_cap_mw'))

    results.set_value(row_num, 'balance_absolute', best_balance)

best_balance
results

feather.write_dataframe(results, 'size_battery_results.feather')
feather.write_dataframe(df, 'size_battery_profiles.feather')

print("Done!")
