# timer wrapper
def timer(func, *args, **kwargs):
    import time
    def warp(*args, **kwargs):
        
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        
        elapsed_minute, elapsed_sec = calculate_time(start_time, end_time)
        if elapsed_minute > 0 and elapsed_sec > 0:
            print("|__ Elapsed time: {}m {}s.".format(elapsed_minute, elapsed_sec))
        else:
            print("|__ Elapsed time: {}s.".format(elapsed_sec))
            
    return warp

def calculate_time(start_time, end_time):
    time = int(end_time-start_time)
    minute = int(time//60)
    sec = int((time%60))
    return minute, sec

# log wrapper for class
def logger(func, *args):
    def warp(*args):
#         verbose = args[0].verbose
        
        if func.__name__ == 'start_trainning_process':
            print('@@@ DEBUG: start <{}>'.format(func.__name__))
            func(*args)
            print('@@@ DEBUG: finish <{}>'.format(func.__name__))
            
        elif func.__name__ == 'run_epoch':
            average_loss = func(*args)
            print('[Epoch {}], train_loss: {:.4f}'.format(args[1], average_loss))
            return average_loss
            
        elif func.__name__ == 'evaluation':
            print('--- DEBUG: start <{}>'.format(func.__name__))
            acc, loss = func(*args)
            print('acc: {:.4f}, loss: {:.4f}'.format(acc, loss))
            return acc, loss
            
        else:
            func(*args)
            
    return warp
