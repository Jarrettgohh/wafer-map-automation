# wafer-map-automation
Automation for wafer mapping -- Jonathan's project

# Documentation
- This would be the documentation on how to use the computer automation for wafer mapping.
- The settings for the automation could set in the `config.json` file

- The template given below would be for the `config.json`:

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
      "rows": [ , ],
      "columns": {
        "X_axis": "",
        "Y_axis": ""
      }
    },
    "area_fraction": {
      "rows": [ , ],
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


# What does the fields in the `config.json` mean?
1. `html_file_directory`: The file directory to the HTML file where the images are stored
2. `images_directory`: The file directory to store the images extracted from *html_file_directory*

*`wafer_mapping_configurations`*:
1.  `folder_directory`: The folder directory where the excel file (to plot the scatter graph) is kept
2.  `file_name`: The name of the excel file to plot the scatter graph

*`wafer_information`*:
1.  `number_of_wafer_points`: Number of wafer points present on the wafer sample
2.  `wafer_ids`: An array of wafer IDs. These IDs would be used as the sheet name in the excel file

*`error_information`*:
- Details regarding the color fills if the automation fails to read the details from the image data
1. `cell_color`: Color to fill the cell with if the automation fails to read the details from the image data
2. `scatter_site_color`: Color to fill the site on the scatter graph, if the automation fails to read the details from the image data

*`to_plot`*:
- Details regarding the row and columns to get the data from, when plotting the scatter graph
1. `rows`: Rows to read the data from
2. `columns`: The column information to read the data from; corresponding to the X and Y axis


*`area_fraction`*:
- Details regarding the row and columns to get the data from, when coloring the points on the scatter graph according to the area fraction
1. `rows`: Rows to read the data from
2. `columns`: The column information to read the data from; corresponding to the area fraction and area fraction percentage


*`color_indicators`*:
- Details regarding the color to fill, corresponding to each area fraction range

- An example of the `config.json`:

``` json
{
  "html_file_directory": "C:/Users/JONATHAN TAN/Desktop/NYP/Reports/Wafer Mapping/20210917b SEM analysis.html",
  "images_directory": "./images",
  "wafer_mapping_configurations": {
    "folder_directory": "C:/Users/gohja/Desktop/wafer-map-automation",
    "file_name": "wafer-mapping-automation-test - Copy.xlsx",
    "wafer_information": {
      "number_of_wafer_points": 49,
      "wafer_ids": [
        "1st Wafer (ML 1)",
        "2nd Wafer (ML 10)",
        "3rd Wafer (ML 12)",
        "4th Wafer (ML 14)",
        "5th Wafer (ML 15)",
        "6th Wafer (ML 2)",
        "7th Wafer (ML 5)",
        "8th Wafer (ML RT 10)",
        "9th Wafer (ML RT 12)",
        "10th Wafer (ML RT 13)",
        "11th Wafer (ML RT 2)",
        "12th Wafer (ML RT 3)",
        "13th Wafer (ML RT 7)"
      ]
    },
    "error_information": {
      "cell_color": "800000", // To convert from text to HEX
      "scatter_site_color": null
    },
    "to_plot": {
      "rows": [2, 50],
      "columns": {
        "X_axis": "B",
        "Y_axis": "C"
      }
    },
    "area_fraction": {
      "rows": [2, 50],
      "columns": {
        "area_fraction": "D",
        "area_fraction_percentage": "E"
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

