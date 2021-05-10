#  Parse the deformation ARAP C/C++ to Python
```
cmake .
swig -c++ -python deformation.i
make
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Alternatively, you can export the PYTHONPATH in the last line of ```~/.bashrc``` in order to avoid export PYTHONPATH multiple times. Example:

```
export PYTHONPATH=$PYTHONPATH:<path_to_module>
```
