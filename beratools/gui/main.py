import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.gui.bt_gui_main import runner

if __name__ == "__main__":
    runner()
