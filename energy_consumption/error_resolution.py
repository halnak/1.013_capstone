import pandas as pd
from weather_scraper import get_sel

'''
For resolving web scraping errors, including:
1. Missing days (this is due to crashing, easy to run the scrape again for those days)
2. Missing hours (this is due to true missing data, will use several different methods to handle this below)
    a. Leave out
    b. Take average between surrounding data
    c. Scale from a previous year's data for that day
    d. Etc.
3. Remove extraneous data
4. Choose a best method for part 2 (doing leave out for now, unless its truly detrimental)
5. Finalize a full csv
'''

month_err = {
    'name': 'month',
    2: {12, 13, 22, 26},
    3: {2, 5, 4},
    7: {31},
    10: {21, 29, 23, 22},
    12: {9}
}

hour_err = {
    'name': 'hour',
    3: {24},
    5: {8},
    6: {9},
    8: {13, 17, 18},
    9: {20},
    10: {20, 24},
    11: {1, 6}
}

time_err = {
    'name': 'time',
    1: {8, 12, 16, 17, 18, 21, 22, 23, 24, 28, 29},
    2: {6, 8, 14, 21, 23, 27},
    3: {15, 17, 18, 24, 28},
    4: {4, 16},
    5: {5, 8, 10, 12},
    6: {1, 8, 9, 17, 18},
    7: {14, 18, 19, 21, 22, 23, 29},
    8: {3, 7, 8, 22, 24, 25},
    9: {1, 2, 6, 7, 8, 9, 13, 14, 18, 20, 22, 24, 27, 28},
    10: {1, 2, 3, 5, 7, 9, 15, 16, 18, 19, 20, 25, 26},
    11: {1, 2, 3, 6, 7, 8, 11, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28},
    12: {2, 5, 6, 8, 10, 27, 28}
}


def convert_times(times):
    new_times = []
    for hour in times:
        if hour == '11:00 PM':
            new_times.append('12:00 AM')
        elif hour[:2] == '11' and hour[-2:] == 'AM':
            new_times.append('12' + hour[2:-2] + 'PM')
        elif hour[:2] == '12':
            new_times.append('1' + hour[2:])
        else:
            spl = hour.split(":")
            h = str(int(spl[0])+1)
            new_times.append(h + hour[len(h):])

    return new_times

def mine_errors(err):
    year = 2022

    base_url = f'https://www.wunderground.com/history/daily/eh/laayoune/GMML/date/2022-'
    df = {
        'date': [],
        'time': [],
        'temp': [],
        'hum': [],
        'wind': [],
        'condition': []
    }
    for month in err:
        if month == 'name':
            continue
        for day in err[month]:
            daily_url = base_url + f'{month}-{day}'
            try:
                times, temps, hums, winds, conds = get_sel(daily_url)
            except:
                print(f'Stopped on day {day} of month {month}')
                continue

            try:
                if times[0] == '11:00 PM':
                    df['time'] += convert_times(times)
                else:
                    df['time'] += times
                df['date'] += [f'{month}/{day}/{year}']*len(times)
                df['temp'] += temps
                df['hum'] += hums
                df['wind'] += winds
                df['condition'] += conds
            except:
                continue
    
    frame = pd.DataFrame(df)
    frame.to_csv(f'laayoune_weather_data_{err["name"]}_errors.csv')
    return df


def combine_all():
    raise Exception('Not implemented yet')


def main():
    # year = 2022
    mine_errors(time_err)


if __name__ == "__main__":
    main()
