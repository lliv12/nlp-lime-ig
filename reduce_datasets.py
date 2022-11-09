import json
import os
import re

output_dir = "/datasets/reviews/"

MAX_ENTRIES_PER_CATEGORY = 3000

files = [
    "appliances-5core.json",
    "arts_crafts_and_sewing-5core.json",
    "automotive-5core.json",
    "books-5core.json",
    "cellphones_and_accessories-5core.json",
    "clothing_shoes_and_jewelry-5core.json",
    "fashion-5core.json",
    "home_and_kitchen-5core.json",
    "movies_and_tv-5core.json",
    "office_products-5core.json",
    "pet_supplies-5core.json",
    "sports_and_outdoors-5core.json",
    "tools_and_home_improvement-5core.json",
    "video_games-5core.json"
]

# Create shorter dataset using the first <max_entries> reviews
def dump_json(file_name, max_entries):
    data = []
    cnt = 0
    for line in open('datasets/reviews_full/' + file_name, 'r'):
        data.append( json.loads(line) )
        if cnt == max_entries:  break
        cnt += 1
    with open(os.getcwd() + output_dir + file_name, 'w') as file:
        file.write( json.dumps(data)[1:-1] )

for f in files:
    dump_json(f, MAX_ENTRIES_PER_CATEGORY)