# -*- coding: utf-8 -*-

import os
import sys

def entrypoint_exists(entry_point):
    executable_dir = os.path.dirname(sys.executable)
    return os.path.exists(os.path.join(executable_dir, entry_point))