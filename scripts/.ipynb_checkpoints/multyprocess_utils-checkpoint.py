import numpy as np
from multiprocessing import Process, Queue
from time import time, sleep


class MultuProcCalc:
    def __init__(self, algorithm, num_workers=1):
        self.algo = algorithm
        self.num_workers = num_workers
            
    def proc(self, out_queue, process_data):
        out_vol = self.algo.binarize(process_data["edges"])
        process_data.update({"out_vol" : out_vol})
        out_queue.put(process_data) 


    def run(self): #-> bin_vol : np.array
        bin_vol = np.zeros_like(self.algo.vol)
        
        out_queue = Queue()
        procs = []
        process_data = self.generate_process_data()
        
        for i in range(self.num_workers):
            p = Process(target = self.proc, args = (out_queue, process_data[i]))  
            p.start()
            procs.append(p)

        for i in range(self.num_workers):
            process_data = out_queue.get()
            e = process_data["edges"]
            bin_vol[e[0][0]:e[0][1],
                    e[1][0]:e[1][1],
                    e[2][0]:e[2][1]] = process_data["out_vol"]

        for p in procs:
            p.join()
        
        
        return(bin_vol)
    
    def generate_process_data(self):
        assert(self.num_workers in (1, 4, 8))
        
        v_s = self.algo.vol.shape
        process_data = []

        if self.num_workers==1:
            process_data.append({"edges" : [[0, v_s[0]],
                                            [0, v_s[1]],
                                            [0, v_s[2]]]})

        if self.num_workers==4:
            process_data.append({"edges" : [[0, v_s[0]//2],
                                            [0, v_s[1]//2],
                                            [0, v_s[2]]]})
            process_data.append({"edges" : [[v_s[0]//2, v_s[0]],
                                            [0, v_s[1]//2],
                                            [0, v_s[2]]]})
            process_data.append({"edges" : [[0, v_s[0]//2],
                                            [v_s[1]//2, v_s[1]],
                                            [0, v_s[2]]]})
            process_data.append({"edges" : [[v_s[0]//2, v_s[0]],
                                            [v_s[1]//2, v_s[1]],
                                            [0, v_s[2]]]})

        if self.num_workers==8:
            process_data.append({"edges" : [[0, v_s[0]//2],
                                            [0, v_s[1]//2],
                                            [0, v_s[2]//2]]})
            process_data.append({"edges" : [[v_s[0]//2, v_s[0]],
                                            [0, v_s[1]//2],
                                            [0, v_s[2]//2]]})
            process_data.append({"edges" : [[0, v_s[0]//2],
                                            [v_s[1]//2, v_s[1]],
                                            [0, v_s[2]//2]]})
            process_data.append({"edges" : [[v_s[0]//2, v_s[0]],
                                            [v_s[1]//2, v_s[1]],
                                            [0, v_s[2]//2]]})    
            process_data.append({"edges" : [[0, v_s[0]//2],
                                            [0, v_s[1]//2],
                                            [v_s[2]//2, v_s[2]]]})
            process_data.append({"edges" : [[v_s[0]//2, v_s[0]],
                                            [0, v_s[1]//2],
                                            [v_s[2]//2, v_s[2]]]})
            process_data.append({"edges" : [[0, v_s[0]//2],
                                            [v_s[1]//2, v_s[1]],
                                            [v_s[2]//2, v_s[2]]]})
            process_data.append({"edges" : [[v_s[0]//2, v_s[0]],
                                            [v_s[1]//2, v_s[1]],
                                            [v_s[2]//2, v_s[2]]]})
        return(process_data)
