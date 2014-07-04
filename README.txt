DESCRIPTION

PiSvM is a parallel Support Vector Machine (SVM) implementation. 
It supports C-SVC, nu-SVC, epsilon-SVR and nu-SVR and has a 
command-line interface similar to the popular LibSVM package.

This is a fork of pisvm-1.2 found at http://pisvm.sourceforge.net/
that aims to increase scalability.

DOCUMENTATION

Documentation on how to use the PiSvM software as well as
performance results are given on the web under:
http://pisvm.sourceforge.net/

Since the command line interface is almost identical to LibSVM,
for general usage instructions concerning SVM training visit:
http://www.csie.ntu.edu.tw/~cjlin/libsvm/

COPYRIGHT/LICENSE

The parallel SMO solver and command line interface is based
on LibSVM code. Therefore the conditions given in COPYRIGHT-LibSVM
apply to the parts of the LibSVM code, that is svm.h, svm.cpp, 
pisvm-train.c pisvm-predict.c and pisvm-scale.c. All of the
pisvm-*.c files contain changes made to enable parallel training.

All of the svm_* files contain code extracted from the svm.cpp
file to increase readability and therefor the conditions in
COPYRIGHT-LibSVM also apply to these files.

All other source code, including the parallel LOQO solver in
subdirectory loqo and the parallel SMO solver in subdirectory
psmo is licensed under the terms of the GPL as given in LICENSE.txt
