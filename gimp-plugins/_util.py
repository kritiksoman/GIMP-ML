import os
import sys

from gimpfu import pdb

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


class tqdm_as_gimp_progress:
    def __init__(self, default_desc=None):
        self.default_desc = default_desc

    def __enter__(self):
        from tqdm import tqdm
        self.tqdm_display = tqdm.display

        def custom_tqdm_display(tqdm_self, *args, **kwargs):
            tqdm_info = tqdm_self.format_dict.copy()
            tqdm_info["prefix"] = tqdm_info["prefix"] or self.default_desc
            tqdm_info["bar_format"] = "{desc}: " if tqdm_info["prefix"] else ""
            # Removed {percentage:3.0f}% from bar_format
            tqdm_info["bar_format"] += "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            pdb.gimp_progress_set_text(tqdm_self.format_meter(**tqdm_info))
            if tqdm_info["total"]:
                pdb.gimp_progress_update(tqdm_info["n"] / float(tqdm_info["total"]))
            else:
                pdb.gimp_progress_pulse()
            self.tqdm_display(tqdm_self, *args, **kwargs)

        tqdm.display = custom_tqdm_display
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        from tqdm import tqdm
        tqdm.display = self.tqdm_display

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
