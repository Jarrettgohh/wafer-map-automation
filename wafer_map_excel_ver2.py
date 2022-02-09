import json
import os
import sys
import pandas as pd

from openpyxl.utils.cell import column_index_from_string
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.marker import Marker
from openpyxl import load_workbook
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)

from append_df_to_excel import append_df_to_excel

f = open('config.json')
config_json = json.load(f)

wafer_mapping_configurations = config_json['wafer_mapping_configurations']

file_directory = wafer_mapping_configurations['file_directory']
file_name = wafer_mapping_configurations['file_name']
sheet_name = wafer_mapping_configurations['sheet_name']

to_plot = wafer_mapping_configurations['to_plot']
to_plot_rows = to_plot['rows']
to_plot_columns = to_plot['columns']

area_fraction = wafer_mapping_configurations['area_fraction']
area_fraction_rows = area_fraction['rows']
area_fraction_columns = area_fraction['columns']

color_indicators = wafer_mapping_configurations['color_indicators']

#
# Open the excel workbook
#

file_path = f'{file_directory}/{file_name}'

wb = load_workbook(filename=file_path)
ws = wb[sheet_name]


def write_area_fraction_to_excel(area_fraction_df: pd.DataFrame):
    try:
        to_write_col = column_index_from_string(area_fraction_columns[0]) - 1

        append_df_to_excel(
            filename=file_path,
            df=area_fraction_df,
            sheet_name=sheet_name,
            startrow=area_fraction_rows[0] - 1,
            startcol=to_write_col,
        )

    except PermissionError:
        print(
            f'Failed to write excel file at path: {file_path}. Do ensure that the file is not open/running.'
        )
        sys.exit()

    except Exception as e:
        print(
            f'Something went wrong with writing to the excel file at path: {file_path}'
        )
        print(e)

        sys.exit()


def plot_scatter_graph():
    # Create a scatter chart
    chart = ScatterChart()
    chart.title = "Scatter Chart Automation Test"
    chart.legend = None

    x_col = column_index_from_string(to_plot_columns[0])
    y_col = column_index_from_string(to_plot_columns[1])
    min_row = to_plot_rows[0]
    max_row = to_plot_rows[1]

    for plot_row_index in range(min_row, max_row + 1):

        area_fraction_value = ws[area_fraction_columns[0] +
                                 str(plot_row_index)].value
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
        wb.save(file_name)

    except PermissionError:
        print(
            f'Failed to save excel file at path: {file_path}. Do ensure that the file is not open/running.'
        )
        sys.exit()


def wafer_map_excel(area_fraction_df: pd.DataFrame):
    write_area_fraction_to_excel(area_fraction_df)
    # plot_scatter_graph()

    os.startfile(file_path)
