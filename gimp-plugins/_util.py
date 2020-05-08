import os
import sys

baseLoc = os.path.dirname(os.path.realpath(__file__))


def add_gimpenv_to_pythonpath():
    env_path = os.path.join(baseLoc, 'gimpenv/lib/python2.7')
    if env_path in sys.path:
        return
    # Prepend to PYTHONPATH to make sure the gimpenv packages get loaded before system ones,
    # since they are likely more up-to-date.
    sys.path[:0] = [
        env_path,
        os.path.join(env_path, 'site-packages'),
        os.path.join(env_path, 'site-packages/setuptools')
    ]
