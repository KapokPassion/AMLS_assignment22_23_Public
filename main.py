import os,sys


tempdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,tempdir+"\\A1")
sys.path.insert(0,tempdir+"\\A2")
sys.path.insert(0,tempdir+"\\B1")
sys.path.insert(0,tempdir+"\\B2")

import A1.train
import A2.train
import B1.train
import B2.train

import A1.test
import A2.test
import B1.test
import B2.test

#training step
# if models had existed in each folder, you can skip the training step.
# models see the OneDrive link in my report
print('A1 training start')
A1.train.run()
print('A2 training start')
A2.train.run()
print('b1 training start')
B1.train.run()
print('B2 training start')
B2.train.run()

#testing step
print('A1 testing start')
A1.test.run()
print('A2 testing start')
A2.test.run()
print('B1 testing start')
B1.test.run()
print('B2 testing start')
B2.test.run()
