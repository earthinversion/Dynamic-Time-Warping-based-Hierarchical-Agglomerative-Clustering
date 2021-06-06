import time
from datetime import datetime, timedelta

"""
- Utpal Kumar, 
  Institute of Earth Sciences,
  Academia Sinica
  @2020
  
"""


def dec2dt(start):
    """
    Convert the decimal type time array to the date-time type array
    """
    results = []
    for st in start:
        year = int(st)
        rem = st - year
        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) -
                     base).total_seconds() * rem
        )
        results.append(result)
    return results


def dec2dt_scalar(st):
    """
    Convert the decimal type time array to the date-time type array
    """
    year = int(st)
    rem = st - year
    base = datetime(year, 1, 1)
    result = base + timedelta(
        seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
    )
    return result


def toYearFraction(date):
    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


class write_summary:
    def __init__(self, scriptname="not-defined", filename="summary.txt", mode="a"):
        self.mode = mode
        self.filename = filename
        self.fileid = open(self.filename, self.mode)
        self.scriptname = scriptname
        self.fileid.write(
            f"###################### {self.scriptname} ########################\n"
        )
        self.fileid.close()

    def write(self, text):
        with open(self.filename, "a") as ff:
            ff.write(f"{text}\n")
