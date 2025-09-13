import os
os.environ["PYAPP_RUNNING"] = "1"
os.environ["PYAPP_RELATIVE_DIR"] = os.getcwd()

from .cli import main

if __name__ == "__main__":
    main()
