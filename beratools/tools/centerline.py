import time

def centerline(callback, **kwargs):
    for x in kwargs:
        callback(x)

    progress = 0
    for i in range(10):
        time.sleep(1.0)
        callback(r'%{}'.format(progress))
        progress += 10
        callback(time.time())
