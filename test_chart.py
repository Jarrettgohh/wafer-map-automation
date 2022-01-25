from openpyxl import Workbook
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)
from openpyxl.chart.marker import Marker, DataPoint

wb = Workbook()
ws = wb.active

rows = [
    ("Sample", ),
    (1, 10),
    (2, 11),
    (3, 12),
    (4, 13),
    (5, 14),
    (5, 15),
    (6, 16),
    (7, 17),
]

for r in rows:
    ws.append(r)

c = ScatterChart()
xvalues = Reference(ws, min_col=1, min_row=2, max_row=9)
yvalues = Reference(ws, min_col=2, min_row=2, max_row=9)

series = Series(xvalues, yvalues)
c.series.append(series)

# c.add_data(xvalues, yvalues, titles_from_data=True)
# c.title = "Chart with patterns"

# # set a pattern for the whole series
# series = c.series[0]

s = c.series[0]
s.marker = Marker('circle', size=10)
s.graphicalProperties.line.noFill = True

# set a pattern for a 6th data point (0-indexed)
pt = DataPoint(idx=5)
pt.graphicalProperties.solidFill = "800000"
s.dPt.append(pt)

ws.add_chart(c, "C1")
wb.save("pattern.xlsx")