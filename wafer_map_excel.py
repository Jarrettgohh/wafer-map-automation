import json

f = open('config.json')
config_json = json.load(f)

color_indicators = config_json['color_indicators']

print(color_indicators)
