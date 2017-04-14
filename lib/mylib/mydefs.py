import numpy as np
import scipy.io
import scipy.ndimage
import scipy.misc

def file2np(arq, x='s'):
    """
    recebe arquivo string e retorna numpy array
    outro arg: 'i' (inteiro), 'f' (float)
    """

    f = open(arq, 'r')
    f = np.asarray(f.read().split())
    if x == 'i':
        return f.astype(np.int)
    elif x == 'f':
        return f.astype(np.float)
    else:
         return f

def read_matfile(arq):
    """
    recebe matfile, retorna sua leitura
    """

    return scipy.io.loadmat(arq)
    
def img2np(img):
    """
    recebe imagem e retorna imagem em formato numpy
    """
    
    out = scipy.ndimage.imread(img)
    return out

def np2img(imgname, vec):
    """
    recebe vetor numpy e escreve numa imagem
    """

    scipy.misc.imsave(imgname, vec)

