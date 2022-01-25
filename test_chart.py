from openpyxl import Workbook
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)
from openpyxl.chart.shapes import GraphicalProperties
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

marker_colors = [
    "800000",
    "0000FF",
    "800000",
    "0000FF",
    "FF7F7F",
    "800000",
    "800000",
    "FF7F7F",
]

for r in rows:
    ws.append(r)

c = ScatterChart()

for i in range(1, 9):
    xvalues = Reference(ws, min_col=1, min_row=i)
    yvalues = Reference(ws, min_col=2, min_row=i)

    series = Series(xvalues, yvalues)
    series.marker = Marker(
        'circle',
        size=15,
        spPr=GraphicalProperties(solidFill=marker_colors[i - 1]))
    series.graphicalProperties.line.noFill = True

    c.series.append(series)

ws.add_chart(c, "C1")
wb.save("scatter.xlsx")