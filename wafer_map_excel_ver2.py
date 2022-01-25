import json
import os
import sys

from openpyxl.utils.cell import column_index_from_string
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.marker import Marker, DataPoint
from openpyxl.drawing.fill import PatternFillProperties, ColorChoice
from openpyxl import load_workbook
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)

f = open('config.json')
config_json = json.load(f)

wafer_mapping_configurations = config_json['wafer_mapping_configurations']
to_plot = wafer_mapping_configurations['to_plot']
area_fraction = wafer_mapping_configurations['area_fraction']
area_fraction_column = area_fraction['column']
color_indicators = wafer_mapping_configurations['color_indicators']

area_fraction

to_plot_rows = to_plot['rows']
to_plot_columns = to_plot['columns']

filename = 'wafer-mapping-automation-test.xlsx'
file_dir = f'C:/Users/gohja/Desktop/wafer-map-automation/{filename}'

#
# Open the excel workbook
#

wb = load_workbook(filename=filename)
ws = wb['13th Wafer (ML RT 7)']

# Create a scatter chart
chart = ScatterChart()
chart.title = "Scatter Chart Automation Test"
chart.legend = None

x_col = column_index_from_string(to_plot_columns[0])
y_col = column_index_from_string(to_plot_columns[1])
min_row = to_plot_rows[0]
max_row = to_plot_rows[1]

for plot_row_index in range(min_row, max_row + 1):

    area_fraction_value = ws[area_fraction_column + str(plot_row_index)].value
    area_fraction_percentage = area_fraction_value * 100

    color = 'blue'

    for color_indicator in color_indicators:
        color_indicator_range = color_indicators[color_indicator]
        color_indicator_lower_range = color_indicator_range[0]
        color_indicator_upper_range = color_indicator_range[1]

        is_area_fraction_percentage_in_range = area_fraction_percentage >= color_indicator_lower_range and area_fraction_percentage <= color_indicator_upper_range

        if is_area_fraction_percentage_in_range:
            color = color_indicator
            break

    xvalues = Reference(ws, min_col=x_col, min_row=plot_row_index)
    yvalues = Reference(ws, min_col=y_col, min_row=plot_row_index)

    # Plot the points on the scatter chart
    series = Series(xvalues, yvalues)
    series.marker = Marker('circle',
                           size=15,
                           spPr=GraphicalProperties(solidFill=color))
    series.graphicalProperties.line.noFill = True

    chart.series.append(series)

ws.add_chart(chart, "G14")

try:
    wb.save(filename)

except PermissionError:
    print('Failed to save file. Ensure file is not open.')
    sys.exit()

os.startfile(file_dir)
