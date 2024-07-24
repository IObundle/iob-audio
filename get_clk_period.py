import numpy as np
import matplotlib.pyplot as plt

from iob_audio import tone, step, impulse, tplot, fplot, thdn
from scipy import signal
from scipy.signal import remez, freqz
    

# create main function
def main():

    fclk = 100e6
    Tclk = 1/fclk
    
    fsin = 48000  # Sampling frequency
    fsout = 44100  # Sampling frequency

    duration = 1  # Duration of the signal in seconds

    #down-sampling rate R floor
    R = int(fclk/fsin)
    
    #window size 
    window_size = 4096


    #create a square wave of frequency fsin, and sample it at fclk
    ns = int(duration*fclk)
    t = np.arange(0, ns*Tclk, Tclk)
    fsyncin = signal.square(2 * np.pi * fsin * t)


    j = 0
    yin = np.ones(int(duration*fsin))
    #print length of yin
    print("length of yin: ", len(yin))

    #print R and length of t
    print("R = ", R)
    print("length of t = ", len(t))
    


    #fsin story

    
    #print the first 10 values of fsyncin
    print("fsyncin = ", fsyncin[0:10])
    #print the last 10 values of fsyncin
    print("fsyncin = ", fsyncin[-10:])

    
    #create vector yin, which is the array of the periods of fsyncin
    last = 0
    i = R
    j = 0
    while i < len(fsyncin):
        if fsyncin[i] > fsyncin[i-1]:
            yin[j] = i - last
            last = i
            i += R
            j += 1
        else:
            i += 1
            

    #remove zeros in yin
    yin = yin[yin != 1]
    

    #print the length of yin
    print("length of yin = ", len(yin))
    
            
    #print the first 10 values of yin
    print("yin = ", yin[0:10])
    #print the last 10 values of yin
    print("yin = ", yin[-10:])
    
    

    #plot
    tplot(yin, fsin, ptitle="yin")
    fplot(yin, fsin, ptitle="yin")
    thdn(yin, fsin, "yin")

    #print the average, max and min
    print("Ave: ", np.mean(yin))
    print("Max: ", np.max(yin[10:]))
    print("Min: ", np.min(yin[10:]))


    #create a vector with the moving average of yin with window size
    zin = np.convolve(yin, np.ones(window_size)/window_size, mode='valid')
    

    #print the length of zin
    print("length of zin = ", len(zin))
    #print the first 10 values of zin
    print("zin = ", zin[0:10])

    #print the average, max and min of the output signal
    print("Ave: ", np.mean(zin))
    print("Max: ", np.max(zin))
    print("Min: ", np.min(zin))
    
    #plot the output signal
    tplot(zin, fsin, ptitle="zin")
    fplot(zin, fsin, ptitle="zin")
    thdn(zin, fsin, "zin")


    #fsout story

    #create a square wave of frequency fsout, and sample it at fclk
    ns = int(duration*fclk)
    t = np.arange(0, ns*Tclk, Tclk)
    fsyncout = signal.square(2 * np.pi * fsout * t)

    yout = np.ones(int(duration*fsin))

    #print the first 10 values of fsyncout
    print("fsyncout = ", fsyncout[0:10])
    #print the last 10 values of fsyncout
    print("fsyncout = ", fsyncout[-10:])

    #create vector yout, which is the array of the periods of fsyncout
    last = 0
    i = R
    j = 0
    while i < len(fsyncout):
        if fsyncout[i] > fsyncout[i-1]:
            yout[j] = i - last
            last = i
            i += R
            j += 1
        else:
            i += 1

    #remove zeros in yout
    yout = yout[yout != 1]

    #print the length of yout
    print("length of yout = ", len(yout))
    #print the first 10 values of yout
    print("yout = ", yout[0:10])
    #print the last 10 values of yout
    print("yout = ", yout[-10:])

    #plot the output signal
    tplot(yout, fsout, ptitle="yout")
    fplot(yout, fsout, ptitle="yout")
    thdn(yout, fsout, "yout")

    #print the average, max and min
    print("Ave: ", np.mean(yout))
    print("Max: ", np.max(yout))
    print("Min: ", np.min(yout))

    #create a vector with the moving average of yout with window size
    zout = np.convolve(yout, np.ones(window_size)/window_size, mode='valid')

    #print the length of zout
    print("length of zout = ", len(zout))
    #print the first 10 values of zout
    print("zout = ", zout[0:10])

    #print the average, max and min of the output signal
    print("Ave: ", np.mean(zout))
    print("Max: ", np.max(zout))
    print("Min: ", np.min(zout))

    #plot the output signal
    tplot(zout, fsout, ptitle="zout")
    fplot(zout, fsout, ptitle="zout")
    thdn(zout, fsout, "zout")



    #create a vector rho with ns samples of fsout/fsin
    rho = np.zeros(ns)
    lin = len(zin)-1
    lout = len(zout)-1
    for i in range(ns):
        rho[i] = zin[min(lin,  int(i*fsin/fclk))] / zout[min(lout, int(i*fsout/fclk))]

    #print the length of rho
    print("length of rho = ", len(rho))
    #print the first 10 values of rho
    print("rho = ", rho[0:10])

    #print the average, max and min of the output signal
    print("Ave: ", np.mean(rho))
    print("Max: ", np.max(rho))
    print("Min: ", np.min(rho))
    
    
    

        
    #plot the output signal
    #tplot(rho, fclk, ptitle="rho")
    #fplot(rho, fclk, ptitle="rho")
    #thdn(rho, fclk, "rho")
    
    

    

    
    

    
    
# call the main function
if __name__ == "__main__":
    main()
    


