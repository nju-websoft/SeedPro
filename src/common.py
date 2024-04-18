import json
import os
from typing import List, Tuple, Dict, Any, Union, Optional


def json_load(path: str):
    if not os.path.isfile(path):
        print('Invalid path: ', path)
        return None
    with open(path, 'r') as fp:
        return json.load(fp)
    
def json_dump(data, path: str, chinese: bool = False):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=not chinese)