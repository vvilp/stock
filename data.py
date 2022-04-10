import urllib
import urllib.request
import time
import datetime
import sqlite3


        
# codes = ['CBA.AX', 'SUN.AX', 'AUDUSD%3DX', '%5EAXJO']
# ALL	Aristocrat Leisure Ltd
# ANZ	Australia and New Zealand Banking Group Ltd
# APT	Afterpay Ltd
# BHP	BHP Group Ltd
# CBA	Commonwealth Bank of Australia
# CSL	CSL Ltd
# FMG	Fortescue Metals Group Ltd
# GMG	Goodman Group
# MQG	Macquarie Group Ltd
# NAB	National Australia Bank Ltd
# NCM	Newcrest Mining Ltd
# REA	REA Group Ltd
# RIO	RIO Tinto Ltd
# TCL	Transurban Group
# TLS	Telstra Corporation Ltd
# WBC	Westpac Banking Corporation
# WES	Wesfarmers Ltd
# WOW	Woolworths Group Ltd
# WPL	Woodside Petroleum Ltd
# XRO	Xero Ltd
# codes = ['ALL.AX', 'ANZ.AX', 'APT.AX', 'BHP.AX', 'CSL.AX', 'FMG.AX', 'GMG.AX']


class DownloadData:
    def __init__(self, code, from_time, to_time):
        self.url = f"https://query1.finance.yahoo.com/v7/finance/download/{code}?period1={from_time}&period2={to_time}&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
        print(self.url)

    def save(self, filename):
        urllib.request.urlretrieve (self.url, filename)

class DB:
    def __init__(self):
        self.con = sqlite3.connect('test.db')

    def create_table(self):
        cur = self.con.cursor()
        cur.execute('''
                        CREATE TABLE IF NOT EXISTS stock (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, 
                            code Text,
                            date datetime,
                            open float,
                            high float,
                            low float,
                            close float,
                            adj_close float,
                            volume float
                        );
                    ''')

codes = ["AUDUSD%3DX", "BHP.AX", "CL%3DF"]
from_time = int(datetime.datetime(2017,1,1).timestamp())
to_time = int(datetime.datetime.now().timestamp())
for code in codes:
    data = DownloadData(code, from_time, to_time)
    data.save(f"{code}.csv")

# db = DB()
# db.create_table()