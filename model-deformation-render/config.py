#!/usr/bin/env python2
"""Configuration values for the project."""
import os.path as path
import click


############################ EDIT HERE ########################################
dir_path = path.dirname(path.realpath(__file__))
SMPL_FP = dir_path + "/SMPL_python_v.1.0.0/smpl"

###############################################################################
# Infrastructure. Don't edit.                                                 #
###############################################################################

@click.command()
@click.argument('key', type=click.STRING)
def cli(key):
    """Print a config value to STDOUT."""
    if key in globals().keys():
        print globals()[key]
    else:
        raise Exception("Requested configuration value not available! "
                        "Available keys: " +
                        str([kval for kval in globals().keys() if kval.isupper()]) +
                        ".")


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter

