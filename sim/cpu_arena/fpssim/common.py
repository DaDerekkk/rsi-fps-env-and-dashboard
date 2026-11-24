import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = json.load(open(os.path.join(HERE, 'catalog.json')))   # frozen table copied from the guide
WEAPONS = {w['id']: w for w in CATALOG['weapons']}
EQUIP = CATALOG['equipment']; UTIL = CATALOG['utilities']; KNIFE = CATALOG['knife']
PHYS_HZ = 60; DECISION_TICKS = 3; DT = 1.0 / PHYS_HZ
BUY_TICKS = 360; LIVE_TICKS = 6900; POST_TICKS = 300; BUY_DEADLINE = 1860
PLANT_TICKS = 180; DEFUSE_KIT = 300; DEFUSE_NOKIT = 600; FUSE_TICKS = 2400
DEFUSE_RANGE = 2.8; PICKUP_RANGE = 2.3; DROP_DELAY = 90
MAP_IDS = ['dust2-cs2-v54', 'mirage-cs2-v54', 'borgo-r3-v54', 'molo-r3-v54', 'nevaio-r3-v54']
DATA = os.environ.get('FPS_DATA', './fps_data/data')
