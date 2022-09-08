import os


def create_dirs(dirs):
    """Create directory recursively

    Args:
        dirs (str): Path to directory
    """
    try:
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        return 0
    except Exception as err:
        print("Error Message: {0}".format(err))
        exit(-1)
