import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'


from gimpfu import *
import sys

activate_this = os.path.join(baseLoc, 'gimpenv', 'bin', 'activate_this.py')
with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))
sys.path.extend([baseLoc])

import shutil
import syncWeights 

def update(flag) :
    gimp.progress_init("Updating plugins...")
    for filename in os.listdir(baseLoc):
        file_path = os.path.join(baseLoc, filename)
        try:
            if os.path.isfile(file_path) and not file_path.endswith('update.py'):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and not (file_path.endswith('weights') or file_path.endswith('gimpenv')) :
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
#     os.system("cd "+baseLoc+";git fetch;git checkout .")
    syncWeights.syncGit(baseLoc)
    os.system("cd "+baseLoc+";chmod +x *.py")
    if flag:
        syncWeights.sync(baseLoc+'weights',flag)
    # pdb.gimp_message("Update Completed Successfully!")
    return

register(
    "update",
    "update",
    "update",
    "Kritik Soman",
    "Your Name",
    "2020",
    "update...",
    "",     
    [(PF_BOOL, "wUpdate", "Update weights", True)],
    [],
    update, menu="<Image>/Layer/GIML-ML")

main()



