import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

days_in_month = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

all_hours = {
    '12:00 AM', '1:00 AM', '2:00 AM', '3:00 AM', '4:00 AM', '5:00 AM', '6:00 AM', '7:00 AM', '8:00 AM', '9:00 AM', '10:00 AM', '11:00 AM',
    '12:00 PM', '1:00 PM', '2:00 PM', '3:00 PM', '4:00 PM', '5:00 PM', '6:00 PM', '7:00 PM', '8:00 PM', '9:00 PM', '10:00 PM', '11:00 PM'
}


def get_sel(url):
    driver = webdriver.Chrome()
    driver.get(url)

    times = []
    temps = []
    hums = []
    winds = []
    conds = []

    try:
        for i in range(1, 50):
            time_el = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//table/tbody/tr[" + str(i) + "]/td[1]/span"))
            )
            temp = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//table/tbody/tr[" + str(i) + "]/td[2]/lib-display-unit/span/span"))
            ) 
            humidity = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//table/tbody/tr[" + str(i) + "]/td[4]/lib-display-unit/span/span"))
            ) 
            wind = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//table/tbody/tr[" + str(i) + "]/td[6]/lib-display-unit/span/span"))
            ) 
            condition = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//table/tbody/tr[" + str(i) + "]/td[10]/span"))
            ) 
            print(f'Time: {time_el.text}, Temperature: {temp.text}, Humidity: {humidity.text}, Wind Speed: {wind.text}, Condition: {condition.text}')
            # time.sleep(10)

            times.append(time_el.text)
            temps.append(temp.text)
            hums.append(humidity.text)
            winds.append(wind.text)
            conds.append(condition.text)
    except:
        print(f'Failed on {i}')
        print(times, temps, hums, winds, conds)
        return times, temps, hums, winds, conds

    driver.close()

    # print(times, temps, hums, winds, conds)
    return times, temps, hums, winds, conds


def get_monthly_data(year, month):
    base_url = f'https://www.wunderground.com/history/daily/eh/laayoune/GMML/date/{year}-{month}-'
    df = {
        'date': [],
        'time': [],
        'temp': [],
        'hum': [],
        'wind': [],
        'condition': []
    }
    for day in range(1, days_in_month[month]+1):
        daily_url = base_url + f'{day}'
        try:
            times, temps, hums, winds, conds = get_sel(daily_url)
        except:
            print(f'Stopped on day {day} of month {month}')
            continue
        df['date'] += [f'{month}/{day}/{year}']*len(times)
        df['time'] += times
        df['temp'] += temps
        df['hum'] += hums
        df['wind'] += winds
        df['condition'] += conds
    
    frame = pd.DataFrame(df)
    frame.to_csv(f'laayoune_weather_data_month{month}.csv')
    return df


def get_year_data(year):
    for i in range(1, 13):
        df = get_monthly_data(year, i)
        print(df)
    return


def verification_log(year):
    for month in range(1, 13):
        df = pd.read_csv(f'energy_consumption/monthly_data_reformatted/laayoune_weather_data_month{month}_fixed.csv')
        # print(df)
        rows, _ = df.shape

        days = set()
        hours = set()
        cur_day = str(1)
        day_ind = 0

        for i in range(rows):
            day = df.at[i, 'date']
            day_ind += 1

            day = day.split("/")[1]

            if day != cur_day:
                if hours != all_hours:
                    missing_hours = all_hours - hours
                    print(f'Day {cur_day} of month {month} missing hours {missing_hours}')
                cur_day = day
                days.add(day)
                hours = set()
                day_ind = 1

            hour = df.at[i, 'time']
            hours.add(hour)

        if len(days) != days_in_month[month]:
            correct = set([str(i) for i in range(1, days_in_month[month]+1)])
            print(f'Month {month} is missing days {correct - days}')
        days = set()


def reformat_date(year):
    for month in range(1, 13):
        df = pd.read_csv(f'energy_consumption/monthly_data_og/laayoune_weather_data_month{month}.csv')
        df = df.loc[:, 'date':]
        rows, _ = df.shape

        for i in range(rows):
            day = df.at[i, 'date']

            if len(day.split("/")) < 3:
                day = day.split("/")[0]
                df.at[i, 'date'] = f'{month}/{day}/{year}'
            else:
                day = day.split("/")[1]
        
        df.to_csv(f'laayoune_weather_data_month{month}_fixed.csv')


def main():
    year = 2022
    # get_year_data(2022)
    verification_log(year)
    # reformat_date(year)


if __name__ == "__main__":
    main()
