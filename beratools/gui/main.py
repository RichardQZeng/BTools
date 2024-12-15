import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())
    
from beratools.gui.bt_gui_main import runner

if __name__ == "__main__":
    runner()
