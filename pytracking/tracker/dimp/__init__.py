#from .dimp import DiMP
from .dimptranst import DiMP
#from .dimptranst_LTMU import DiMP_LTMU
def get_tracker_class():
    return DiMP
    # return DiMP_LTMU