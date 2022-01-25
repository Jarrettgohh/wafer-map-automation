import json
from openpyxl.chart.marker import Marker
from openpyxl import load_workbook
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)

f = open('config.json')
config_json = json.load(f)

color_indicators = config_json['color_indicators']

filename = 'wafer-mapping-automation-test.xlsx'
file_dir = f'C:/Users/gohja/Desktop/wafer-map-automation/{filename}'

wb = load_workbook(filename=filename)
ws = wb.active

chart = ScatterChart()
chart.title = "Scatter Chart Automation Test"
chart.legend = None

xvalues = Reference(ws, min_col=2, min_row=2, max_row=50)
yvalues = Reference(ws, min_col=3, min_row=2, max_row=50)

series = Series(xvalues, yvalues)
series.marker = Marker('circle', size=15)
series.graphicalProperties.line.noFill = True
chart.series.append(series)

ws.add_chart(chart, "G14")
wb.save(filename)