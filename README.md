# wafer-map-automation
Automation for wafer mapping -- Jonathan's project

``` json
{
  "html_file_directory": "",
  "images_directory": "",
  "wafer_mapping_configurations": {
    "file_directory": "",
    "file_name": "",
    "wafer_information": {
      "number_of_wafer_points": ,
      "wafer_ids": [ ]
    },
    "error_information": {
      "cell_color": "",
      "scatter_site_color": ""
    },
    "to_plot": {
      "rows": [*<start_row>*, *<end_row>*],
      "columns": {
        "X_axis": "",
        "Y_axis": ""
      }
    },
    "area_fraction": {
      "rows": [*<start_row>*, *<end_row>*],
      "columns": {
        "area_fraction": "",
        "area_fraction_percentage": ""
      }
    },
    "color_indicators": {
      "4472C4": [0, 5],
      "8EA9DB": [5, 15],
      "FFFFFF": [15, 25],
      "FF6D6D": [25, 35],
      "FF0000": [35, 40]
    }
  }
}
```

