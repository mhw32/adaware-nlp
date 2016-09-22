import os

def confirm_params_override(files, ensure_prompt=False):
    will_override = False
    project_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    project_base = os.path.abspath(project_base)
    for f in files:
        if os.path.isfile(os.path.join(project_base, f)):
            will_override = True
            break
    
    if will_override or ensure_prompt:
        print '='*30
        print "WARNING, running this file will override at least one of the following:"
        print "{}".format("\n".join(files))
        
        confirm = raw_input("Would you like to proceed? [Y/n]: ")
        if confirm != 'Y':
            raise RuntimeError('Please enter Y if you would like to proceed. Try harder next time')
