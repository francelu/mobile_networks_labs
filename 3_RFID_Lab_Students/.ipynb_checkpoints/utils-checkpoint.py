import numpy as np
import pandas as pd
import re


def read_data(file):
    data = {
        'EPC': [],
        'ant': [],
        'count': [],
        'year': [],
        'month': [],
        'day': [],
        'hour': [],
        'minute': [],
        'second': [],
        'nanosecond': [],
        'Frequency': [],
        'Phase': []
    }
    
    pattern = re.compile(
        r"Background Read: EPC:(?P<EPC>\w+) ant:(?P<ant>\d+) count:(?P<count>\d+) time:(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)T(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\.(?P<nanosecond>\d+)-\d+, Frequency : (?P<Frequency>\d+), Phase : (?P<Phase>\d+)"
    )
    
    with open(file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                data['EPC'].append(match.group('EPC'))
                data['ant'].append(int(match.group('ant')))
                data['count'].append(int(match.group('count')))
                data['year'].append(int(match.group('year')))
                data['month'].append(int(match.group('month')))
                data['day'].append(int(match.group('day')))
                data['hour'].append(int(match.group('hour')))
                data['minute'].append(int(match.group('minute')))
                data['second'].append(int(match.group('second')))
                data['nanosecond'].append(int(match.group('nanosecond')))
                data['Frequency'].append(int(match.group('Frequency')))
                data['Phase'].append(int(match.group('Phase')))
    
    df = pd.DataFrame(data)
    return df


