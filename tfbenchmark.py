import tensorflow
import timeit
import numpy
from math import sqrt, cos

tensorflow_times_list = []
for i in range(0, 50):
    global A, B, result
    A = numpy.random.rand(1000000, 1).astype(numpy.float64)
    B = numpy.random.rand(1000000, 1).astype(numpy.float64)
    A = tensorflow.constant(A)
    B = tensorflow.constant(B)

    # Create a session object
    config = tensorflow.ConfigProto()
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8

    sess = tensorflow.Session(config=config)

    global X, Y, Z
    X = tensorflow.cos(A)
    Y = tensorflow.multiply(X, B)
    Z = tensorflow.abs(Y)
    result = tensorflow.sqrt(Z)

    timer = timeit.Timer(
        "sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, X, Y, Z, result")
    tensorflow_times_list.extend(timer.repeat(1, 1))

    res = sess.run(result[0])
    a = sess.run(A[0])
    b = sess.run(B[0])
    res2 = sess.run(result[1000])
    a2 = sess.run(A[1000])
    b2 = sess.run(B[1000])
    sess.close()

    assert(abs(sqrt(abs(cos(a)*b)) - res) < 1e-10)
    assert(abs(sqrt(abs(cos(a2)*b2)) - res2) < 1e-10)

print('timings:', tensorflow_times_list)
print('mean time (in sec):',
      sum(tensorflow_times_list)/len(tensorflow_times_list))
