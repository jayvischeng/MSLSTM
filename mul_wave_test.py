import pywt
import sys
import numpy as np
def Multi_Scale_Wavelet(X_Training0,Y_Training=[],Level=2,Is_Wave_Let=True):
    print("X_Training")
    NN = X_Training0.shape[0]
    print(X_Training0.shape)
    Level2 = Level + 1
    TEMP_XData = [[] for i in range(Level2)]
    wave_let = 'haar'
    X_Training = X_Training0
    if Is_Wave_Let == True:
        for tab in range(Level2):
            X = []
            for each_feature in range(len(X_Training[0])):
                #print(str(each_feature+1)+" is processing"+" length is "+str(len(X_Training[:,each_feature])))
                coeffs = pywt.wavedec(X_Training[:,each_feature],wave_let, mode='zpd', level=Level)
                current_level = Level2 - tab
                print("-------------------------------------------------+++++++++++++++++++++++" + str(tab))
                print(coeffs)
                for tab2 in range(tab+1,Level2+1):
                    try:
                        pass
                        #coeffs[tab2] = None
                    except:
                        pass

                X_WAVELET_REC = pywt.waverec(coeffs, wave_let)
                print("-------------------------------------------------"+str(tab))
                print(X_WAVELET_REC)
                #print("recoverying legnth is "+str(len(X_WAVELET_REC)))
                X.append(X_WAVELET_REC[:NN])
                #print(np.transpose(np.array(X)).shape)

            TEMP_XData[current_level - 1].extend(np.transpose(np.array(X)))

        TEMP_XData = np.array(TEMP_XData)
        #print("11111111111")
        #print(TEMP_XData.shape)
    else:
        for tab in range(Level):
            current_level = Level - tab
            TEMP_XData[current_level - 1].extend(X_Training)

        TEMP_XData = np.array(TEMP_XData)
    #print("22222222222222")
    #print(TEMP_XData)
    #print("33333333333333")
    return  TEMP_XData,X_Training,Y_Training
def waveFn(wavelet):
    if not isinstance(wavelet, pywt.Wavelet):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet

# given a single dimensional array ... returns the coefficients.
def wavedec(data, wavelet, mode='sym'):
    wavelet = waveFn(wavelet)

    dLen = len(data)
    coeffs = np.zeros_like(data)
    #print("max  level is ")
    level = pywt.dwt_max_level(dLen, wavelet.dec_len)
    #print("max  level is "+str(level))

    a = data
    end_idx = dLen
    for idx in xrange(level):
        a, d = pywt.dwt(a, wavelet, mode)
        begin_idx = end_idx/2
        coeffs[begin_idx:end_idx] = d
        end_idx = begin_idx

    coeffs[:end_idx] = a
    return coeffs

def waverec(data, wavelet, mode='sym'):
    wavelet = waveFn(wavelet)

    dLen = len(data)
    level = pywt.dwt_max_level(dLen, wavelet.dec_len)

    end_idx = 1
    a = data[:end_idx] # approximation ... also the original data
    d = data[end_idx:end_idx*2]
    for idx in xrange(level):
        a = pywt.idwt(a, d, wavelet, mode)
        end_idx *= 2
        d = data[end_idx:end_idx*2]
    return a

def fn_dec(arr):
    print(arr)
    coeffs = (pywt.wavedec(arr, 'haar', 'zpd'))
    coeffs[3] = None
    coeffs[2] = None

    bbb = pywt.waverec(coeffs,'haar')
    print(pywt.wavedec(arr, 'haar', 'zpd'))
    print(bbb)
    print(pywt.wavedec(bbb, 'db1', 'zpd',1))
    print(pywt.waverec(coeffs,'haar'))


    return np.array(map(lambda row: reduce(lambda x,y : np.hstack((x,y)), pywt.wavedec(row, 'haar', 'zpd')), arr))
    # return np.array(map(lambda row: row*2, arr))

if __name__ == '__main__':
    test  = -1
    np.random.seed(10)
    wavelet = waveFn('haar')
    if test==0:
        # SIngle dimensional test.
        a = np.random.randn(1,8)
        print "original values A"
        print a[0]
        print "decomposition of A by method in pywt"
        print fn_dec(a[0])
        print " decomposition of A by my method"
        coeffs =  wavedec(a[0], 'haar', 'zpd')
        print coeffs
        print "recomposition of A by my method"
        print waverec(coeffs, 'haar', 'zpd')
        sys.exit()
    if test==1:
        a = np.random.randn(16,16)
        # 2 D test
        print "original value of A"
        print a

        # decompose the signal into wavelet coefficients.
        dimensions = a.shape
        for dim in dimensions:
            a = np.rollaxis(a, 0, a.ndim)
            ndim = a.shape
            #a = fn_dec(a.reshape(-1, dim))
            print("aaa")
            print(a.reshape(-1, dim))
            a = np.array(map(lambda row: wavedec(row, wavelet), a.reshape(-1, dim)))
            a = a.reshape(ndim)
            print("bbb")
            print(a)
        print " decomposition of signal into coefficients"
        print a

        # re-composition of the coefficients into original signal
        for dim in dimensions:
            a = np.rollaxis(a, 0, a.ndim)
            ndim = a.shape
            a = np.array(map(lambda row: waverec(row, wavelet), a.reshape(-1, dim)))
            a = a.reshape(ndim)
        print "recomposition of coefficients to signal"
        print a

#print(Multi_Scale_Wavelet(a))

#A = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
A = [80,61,75,71,63,59,76,63]
a = pywt.wavedec(A,'haar',level=1)
b = pywt.dwt(A,'haar')
#a[1] = None
#A1 = pywt.waverec(a,'db1')
#a = pywt.wavedec(A,'db1',level=1)
#a[0] = None
#D1 = pywt.waverec(a,'db1')

print(a)
print(b)
#print(D1)
#a = pywt.wavedec(A,'db1',level=1)

#print(a[0])
#print(a[1])

#A0 = pywt.idwt(a[0],a[1],'db1')
#A0 = pywt.downcoef(A,a,'db1')
#print(A0)

#A0 =
