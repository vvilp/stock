import urllib
import urllib.request

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
codes = ['ALL.AX', 'ANZ.AX', 'APT.AX', 'BHP.AX', 'CSL.AX', 'FMG.AX', 'GMG.AX']

for code in codes:
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{code}?period1=1487548800&period2=1645315200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
    urllib.request.urlretrieve (url, f"data/{code}.csv")



