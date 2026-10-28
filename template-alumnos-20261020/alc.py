import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

########### FUNCIONES DEL MODULO ALC ###########

# Traspone una matriz A.
def traspuesta(A):
    return A.T

def traspuesta_bis(A):
    filas, columnas = np.shape(A)
    T = np.zeros((columnas, filas))

    for i in range(filas):
        for j in range(columnas):
            T[j,i] = A[i,j]
    return T

# Calcula Ax = b.
def calcularAx(A,x):
    return A @ x

def calcularAx_bis(A,x):
    n, m = np.shape(A)
    b = np.zeros(n) 

    for i in range(n):
        for j in range(m):
            b[i] += A[i,j] * x[j]
    return b

# Calcula el producto matricial entre A y B.
#def prodMat(A, B):
#    return A @ B

def prodMat(A, B):
    A = np.array(A)
    B = np.array(B)

    if len(A.shape) == 1:
        fA, cA = 1, len(A)
        A2 = np.zeros((fA, cA))
        for j in range(cA):
            A2[0, j] = A[j]
        A = A2

    if len(B.shape) == 1:
        fB, cB = len(B), 1
        B2 = np.zeros((fB, cB))
        for i in range(fB):
            B2[i, 0] = B[i]
        B = B2

    fA, cA = A.shape
    fB, cB = B.shape

    if cA != fB:
        return None

    res = np.zeros((fA, cB))
    for i in range(fA):
        for j in range(cB):
            suma = 0
            for k in range(cA):
                suma += A[i, k] * B[k, j]
            res[i, j] = suma

    return res

# Calcula el producto interno entre dos vectores.
def producto_interno(x, y):
    if len(x) != len(y):
        return None
    suma = 0.0
    for i in range(len(x)):
        suma += x[i] * y[i]
    return suma

def prod_vectorial(u, v):
    u_flat = u.reshape(3)
    v_flat = v.reshape(3)
    
    w0 = u_flat[1] * v_flat[2] - u_flat[2] * v_flat[1]
    w1 = u_flat[2] * v_flat[0] - u_flat[0] * v_flat[2]
    w2 = u_flat[0] * v_flat[1] - u_flat[1] * v_flat[0]
    
    return np.array([w0, w1, w2])

#LABO 1 

# Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64.
def error(x,y):
    x64 = np.float64(x)
    y64 = np.float64(y)
    
    return abs(y64 - x64)


# Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64.
def error_relativo (x,y):
    x = np.float64(x)
    y = np.float64(y)
    if x == 0:
        return np.abs(x - y)
    else:
        return np.abs(x - y)/np.abs(x)
    
# Devuelve True si ambas matrices son iguales y False en otro caso. Las matrices pueden tener distintas dimensiones y/o valores.
def matricesIguales (A, B, atol = 1e-08):
    if np.shape(A)[0] != np.shape(B)[0] or np.shape(A)[1] != np.shape(B)[1]:
        return False
    else:
        for fila in range(np.shape(A)[0]):
            for columna in range(np.shape(A)[1]):
                error = np.abs(np.float64(A[fila, columna]) - np.float64(B[fila, columna]))
                if error > atol:
                    return False
    return True

#LABO 2 

# Recibe un angulo theta y retorna una matriz de 2x2 que rota un vector dado en ese angulo.
def rota(theta):

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

# Recibe una tira de numeros s y retorna una matriz cuadrada de n x n , donde n es el tamaño de s.
# La matriz escala la componente i de un vector de Rn en un factor s[i].
def escala(s):
    n = len(s)
    M = np.zeros((n,n))

    for i in range(n):
        M[i,i] = s[i]
    return M

# Recibe un angulo theta y una tira de numeros s, retorna una matriz de 2x2 que rota el vector 
# en el angulo y luego lo escala en un factor s.
def rota_y_escala(theta, s):

    R = rota(theta)
    E = escala(s)
    M = prodMat(E, R) 
    return M

# Recibe un angulo theta, una tira de numeros s (en R2), y un vector b en R2. Retorna una matriz de 3x3 que 
# rota el vector en un angulo theta, luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b.
def afin(theta,s,b):
  
    RE = rota_y_escala(theta, s)
    M = np.eye(3)
    M[:2, :2] = RE
    M[:2, 2] = b 
    return M

# Recibe un vector v (en R2), un angulo theta, una tira de numeros s (en R2), y un vector b en R2.
# Retorna el vector resultante de aplicar la transformacion afin a v.
def trans_afin(v,theta,s,b):

    M = rota_y_escala(theta, s)
    w = calcularAx(M,v) + b 
    return w

#LABO 3 

# Devuelve la norma p del vector x .
def norma(x, p):
    if p == 'inf':
        return np.max(np.abs(x))
    else:
        return (np.sum(np.abs(x) ** p)) ** (1/p) #formula general
    
# Recibe X, una lista de vectores no vacios y un escalar p. Devuelve una lista donde cada elemento 
# corresponde a normalizar los elementos de X con la norma p.
def normaliza(X, p):
    lista = []
    for x in X:
        if norma(x, p) == 0:
            lista.append(x)
        else:
            lista.append(x / norma(x, p))
    return lista

# Devuelve la norma ||A||_{p,q} y el vector x en el cual se alcanza el maximo.
def normaMatMC(A, q, p, Np):
    A = A.astype(float)
    
    filas, columnas = np.shape(A)
    vmax = [np.zeros(columnas), 0]
    
    X = np.random.randn(Np, columnas)

    for i in range(Np):
        x = X[i]
        Ax = calcularAx(A, x)
        normaAx = norma(Ax, q)
        normax = norma(x, p)
        
        if normax == 0:
            continue
        
        res = normaAx / normax
        
        if res > vmax[1]:
            x = x / normax
            vmax = [x, res]
            
    return vmax[1], vmax[0]

# Devuelve una lista con las normas 1 e infinito de una matriz A usando sus expresiones equivalentes.
# Le agregamos la opcion de solo ingresar uno de los valores predeterminados de p y que devuelva solo esa norma.
def normaExacta(A, p=[1, 'inf']):
    
    A = A.astype(float)
    
    filas, columnas = np.shape(A)
    
    if (p == 1) or (p == 'inf') or (p == [1, 'inf']):
        sumas = []
        for c in range(columnas):
            sumatoria = 0
            for f in range(filas):
                sumatoria += np.abs(A[f,c]) 
            sumas.append(sumatoria)
            
        normauno = np.max(sumas)
        
        sumas = []
        
        for f in range(filas):
            sumatoria = 0
            for c in range(columnas):
                sumatoria += np.abs(A[f,c]) 
            sumas.append(sumatoria)
            
        normainf = np.max(sumas)
        if (p == [1, 'inf']):
            return [normauno, normainf]
        elif p == 1:
            return normauno
        else:
            return normainf
    
    else:
        return None

# Devuelve el numero de condicion de A usando la norma inducida p.
def condMC(A, p):
    if A is None:
        return None
    
    A = A.astype(float)
    
    f, c = A.shape
    
    if f != c:
        return None
    
    invA = inversa(A)
    
    normaA = normaMatMC(A, p, p, 1000)
    normaInv = normaMatMC(invA, p, p, 1000)
    
    return normaA[0] * normaInv[0]

# Que devuelve el numero de condicion de A a partir de la formula: norma p de A por la norma p de su inversa.
def condExacta(A, p):
    if (p == 1) or (p == 'inf'):
        aux = normaExacta(A)
        inv = normaExacta(inversa(A))
    else:
        return None
    
    if p == 1:
        return aux[0] * inv[0]
    elif p == 'inf':
        return aux[1] * inv[1]

#LABO 4 

# Calcula la factorizacion LU de la matriz A. Devuelve las matrices L, U y el numero de operaciones realizadas.
# En caso de que la matriz no pueda factorizarse, retorna None.
def calculaLU (A): 
    if A is None:
        return None, None, 0
    
    A = A.astype(float)

    cant_op = 0
    f, c = A.shape
    Ac = A.copy()
    
    if f != c:
        return None, None, 0
    
    L = np.eye(f)
    U = np.zeros((f,c))
    
    for columna in range(c-1):
        
        pivote = Ac[columna, columna]
        if pivote == 0:
            return None, None, 0
        
        for n in range (columna+1, c, 1):
            Ac[n, columna] = Ac[n, columna] / pivote
            L[n, columna] = Ac[n, columna]
            cant_op += 1
            
            for i in range (columna+1, c, 1):
                Ac[n, i] = Ac[n, i] - (Ac[n, columna] * Ac[columna, i])
                cant_op += 2
                
    for i in range(c):
        for j in range(c):
            if i <= j:
                U[i, j] = Ac[i, j]
    
    return L, U, cant_op

# Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar si es triangular inferior o superior
# usando el argumento inferior (por default asumimos que es triangular inferior).
def res_tri(L, b, inferior=True):
    f, c = L.shape
    
    res = np.zeros(f)
    
    if inferior:
        for i in range(f):
            suma = 0
            for j in range(i):
                suma += L[i, j] * res[j]
            res[i] = (b[i] - suma) / L[i, i]
    else:
        for i in range(f-1, -1, -1):
            suma = 0
            for j in range(i+1, f):
                suma += L[i, j] * res[j]
            res[i] = (b[i] - suma) / L[i, i]

    return res

# Calcula la inversa de A empleando la factorizacion LU y las funciones que resuelven sistemas triangulares.
def inversa(A):
    if A is None:
        return None
    
    A = A.astype(float)
    
    f, c = A.shape
    
    if f != c:
        return None
    
    if determinante(A) == 0:
        return None
    
    L, U, op = calculaLU(A)
    
    I = np.eye(f)
    
    inv = np.zeros([f,c])
    

    for i in range(f):
        y = res_tri(L, I[:, i], inferior=True)
        x = res_tri(U, y, inferior=False)
        inv[:, i] = x
    
    return inv

# Usamos la funcion de numpy ya que es realmente muy costoso usar la version del modulo.
def inversa_bis(A):
    if A is None:
        return None
    
    A = A.astype(float)
    
    f, c = A.shape
    
    if f != c:
        return None

    return np.linalg.inv(A)

# Calcula el determinante de A por regla de las diagonales.
def determinante(A):
    A = A.astype(float)
    
    f, c = A.shape
    
    if f != c:
        return None
    
    L, U, nops = calculaLU(A)
    res = 1
    
    for i in range(c):
        res *= U[i,i]
        
    return res * ((-1)** nops)

# Calcula la factorizacion LDV de la matriz A, de forma tal que A = LDV, con L triangular inferior, D diagonal, 
# y U triangular superior. En caso de que no se pueda factorizar retorna None.
def calculaLDV (A):
    
    A = A.astype(float)
    
    L, U, op = calculaLU(A)
    V, D, op = calculaLU(traspuesta(U))
    
    return L, D, traspuesta(V)

# Revisa si la matriz A es simetrica definida positiva usando la factorizacion LDV.
def esSDP (A, atol=1e-8):
    
    A = A.astype(float)
    
    f, c = A.shape
    
    if f != c:
        return False
    
    if not matricesIguales(A, traspuesta(A), atol):
        return False
    
    L, D, V = calculaLDV(A)
    
    for i in range(len(D)):
        if D[i,i] <= atol:
            return False
    
    return True

# Calcula la factorizacion de Cholesky usando A = L L^t. Devuelve L. 
def calculaCholesky(A, atol=1e-10):
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            suma = 0
            if j > 0:
                suma = np.sum(L[i, :j] * L[j, :j])

            if i == j:
                val = A[i, i] - suma
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - suma) / L[j, j]

    return L
    
#LABO 5 

# Calcula la factorizacion QR de A por Gram Schmidt. Retorna Q, R y el numero de operaciones realizadas.
# Si no aplica, retorna None.
def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    if A.shape[0] < A.shape[1]:
        A = traspuesta(A)

    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    nops = 0

    for j in range(n):
        uj = A[:, j].copy()

        if j > 0:
            nops += j * (m * 2 - 1)

            R[:j, j] = prodMat(traspuesta(Q[:, :j]), uj)
            nops += m * (2) - 1

            uj = uj - prodMat(Q[:, :j], R[:j, j])
            nops += m

        norm_uj = norma(uj,2)
        nops += m * 2 

        if norm_uj < tol:
            return None

        Q[:, j] = uj / norm_uj
        nops += m

        R[j, j] = norm_uj

    if retorna_nops:
        return Q, R, nops
    else:
        return Q, R
    
# Calcula la factorizacion QR de A por reflexiones Householder. Retorna Q, R y el numero de operaciones realizadas.
# Si no aplica, retorna None.
def QR_con_HH(A, tol = 1e-12):
    A = A.astype(float)
    
    f, c = A.shape
    
    #if f != c:
    #    return None
    
    R = A.copy()
    Q = np.eye(f)

    for k in range(c):
        x = R[k:f,k]
        e = np.eye(len(x))
        
        alfa = -np.sign(x[0]) * norma(x, 2)
        u = x - alfa * e[0]
        
        if norma(u, 2) > tol:
            u = u / norma(u, 2)
            
            u_col = np.zeros((len(u), 1)) 
            for i in range(len(u)):
                u_col[i, 0] = u[i]

            uuT = prodMat(u_col, traspuesta(u_col))
            Hk = np.eye(len(u)) - 2 * uuT

            Hk_ = np.eye(f)
            Hk_[k:f,k:f] = Hk
            
            R = prodMat(Hk_, R)
            Q = prodMat(Q, traspuesta(Hk_))
    
    return Q, R

# Recibe A y el metodo a utilizar para calcular la factorizacion QR, opcionalmente la tolerancia. Devuelve Q, R y el numero de operaciones.
# Por defecto usa reflexiones de Householder. Si no aplica, retorna None.
def calculaQR (A, metodo = 'RH', tol = 1e-12):
    A = A.astype(float)
    
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    else:
        return None

#LABO 6 

# Aplica el metodo de la potencia a la matriz A. Permite variar la tolerancia y el máximo de operaciones a realizar.
# Devuelve el autovector asociado v, el mayor autovalor lambda, y el numero de operaciones realizadas k.
def metpot2k (A, tol = 1e-15, K = 1000):
    A = A.astype(float)
    
    f, c = A.shape
    
    if f != c:
        return None
    
    v = np.random.rand(f)
    v = v / norma(v, 2)
    v_ = F(A, v, 2)
    e = producto_interno(v_ , v)
    k = 0
    
    while (np.abs(e-1)) > tol and (k < K) :
        v = v_
        v_ = F(A, v, 2)
        e = producto_interno(v_ , v)
        k += 1
    
    Av = calcularAx(A, v)
    lmbda = producto_interno(v, Av)
    
    return v, lmbda, k

# Aplica la transformacion F para la realizacion del metodo de la potencia. 
def F (A, v, k):
    w = v.copy()
    
    for i in range(k):
        w = calcularAx(A, w)
        n = norma(w, 2)
        if n > 0:
            w = w / n
        else:
            return w
    return w
    
# Admite dos vectores a, b y retorna la matriz M, su producto externo.
def producto_externo(a, b):
    n = len(a)
    m = len(b)
    
    M = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            M[i, j] = a[i] * b[j]
    
    return M

# Realiza la diagonalizacion RH de la matriz A usando el metodo de la potencia y reflexiones de Householder.
# Retorna matriz de autovectores S y matriz de autovalores D tal que A = S D S^t. Si no aplica, devuelve None.
def diagRH(A, tol=1e-15, K=1000):

    n = np.shape(A)[0]

    v1, lambda1, _ = metpot2k(A,tol,K)

    e1 = np.zeros(n)
    e1[0] = 1

    Hv1 = np.eye(n) - 2 * producto_externo((e1-v1),(e1-v1)) / prodMat(e1-v1,e1-v1)

    if n == 2:
        S = Hv1
        D = prodMat(prodMat(Hv1,A),traspuesta(Hv1))
    else:
        B = prodMat(prodMat(Hv1,A),traspuesta(Hv1))
        A_moño = B[1:n,1:n]
        S_moño, D_moño = diagRH(A_moño, tol, K)
        D = np.zeros_like(A)
        D[0, 0] = lambda1
        D[1:, 1:] = D_moño
        S_aux = np.eye(n)
        S_aux[1:, 1:] = S_moño
        S = prodMat(Hv1,S_aux)

    return S, D

#LABO 7 

# Genera una matriz de transicion al azar de n × n normalizada por columnas y con entradas al azar en el intervalo [0,1].
def transiciones_al_azar_continuas(n):
    A = np.random.rand(n, n)
    col_sums = A.sum(axis=0)
    A = A / col_sums
    return A 

# Entra n entero para determinar el tamaño de la matriz y thres como parametro para evaluar los valores de la matriz.
# Si el valor elegido es mayor a thres, se le asigna 0, si no, se le asigna 1. Se normaliza cada columna para que su suma sea 1.
# Retorna una matriz T de transicion al azar de n × n normalizada por columnas y con entradas [0,1] determinadas por thres.
def transiciones_al_azar_uniformes(n, thres):

    T = np.random.random((n,n))
    suma_cols = np.zeros(n)

    for j in range(n):
        for i in range(n):
            if T[i,j] <= thres:
                T[i,j] = 1
            elif T[i,j] > thres:
                T[i,j] = 0

    for j in range(n):
        if np.sum(T[:,j]) == 0: 
            i_rand = np.random.randint(0,n) 
            T[i_rand,j] = 1

    for j in range(n):
        for i in range(n):
            suma_cols[j] += T[i,j]

    T = T / suma_cols

    return T

# Calcula el nucleo de la matriz A diagonalizando la multiplicacion matricial de su matriz traspuesta por A, usando el metodo diagRH.
# El nucleo corresponde a los autovectores de autovalores con modulo menor o igual a tol.
# Retorna los autovectores en cuestion como una matriz de nxk con k el numero de autovectores en el nucleo.
def nucleo(A,tol=1e-15):

    k = 0
    m,n = np.shape(A)
    nucleo = np.zeros(n)

    A_t = traspuesta(A)
    A_tA = prodMat(A_t, A)

    S,D = diagRH(A_tA,tol,K=1000)
    autovalores = np.diag(D)

    for j in range(n):
        if abs(autovalores[j]) <= tol:
            k += 1
            nucleo[j] = 1

    if k == 0:
        return np.zeros((0, 0))

    autovectores = np.zeros((n,k))

    col = 0
    for j in range(n):
            if nucleo[j] == 1:
                autovectores[:,col] = S[:,j]
                col += 1

    return autovectores

# Recibe una lista 'listado' , con tres elementos: lista con indices i , lista con indices j y lista con valores de aij de la matriz A.
# Tambien las dimensiones de la matriz a travez de m filas y n columnas. Los elementos menores a tol se descartan.
# Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero.
# Retorna una lista con un diccionario y una tupla. El diccionario {(i,j): aij} que representa los elementos no nulos de la matriz A.
# Los elementos con modulo menos a tol deben descartarse por default. La tupla (m filas, n columnas) permite conocer las dimensiones de la matriz.
def crea_rala(listado, m_filas ,n_columnas, tol=1e-15):

    if len(listado) == 0:
        return {}, (m_filas, n_columnas)

    dicc = {}

    indices_i = listado[0]
    indices_j = listado[1]
    vals_A_ij = listado[2]

    for i, j, val in zip(indices_i, indices_j, vals_A_ij):
        if abs(val) >= tol:
                dicc[(i,j)] = val

    tupla = (m_filas, n_columnas)

    return dicc, tupla

# Recibe una matriz rala creada con crear_rala y un vector v. Retorna un vector w resultado de multiplicar A con v.
def multiplica_rala_vector(A,v):
   
    A_dict, dims = A
    m, n = dims

    w = np.zeros(m)

    for (i, j), val in A_dict.items():
        w[i] += val * v[j]

    return w

#LABO 8

def normalizarMatriz(A, p):
    m, n = np.shape(A)
    B = np.zeros((m,n))

    for j in range(n):
        col = A[:, j]
        nrm = norma(col, p)
        if nrm == 0:
            B[:, j] = col
        else:
            B[:, j] = col / nrm

    return B

def ordenar_desc(lista):
    indices = list(range(len(lista)))
    
    for i in range(len(indices)):
        max_idx = i
        for j in range(i+1, len(indices)):
            if lista[indices[j]] > lista[indices[max_idx]]:
                max_idx = j
        indices[i], indices[max_idx] = indices[max_idx], indices[i]
    
    return np.array(indices)   

# Realiza la descomposicion SVD de A tal que A = U Σ V^t e incluye como parametro k, para poder tomar solo los primeros valores singulares.
# Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k).
def svd_reducida(A, k="max", tol=1e-15):

    m, n = np.shape(A)
    At = traspuesta(A)

    if m >= n:
        AtA = prodMat(At,A)  

        S, D = diagRH(AtA, tol, K=1000)
        autovalores = np.diag(D).copy()

        indices_ordenados = ordenar_desc(autovalores)

        indices_validos = []
        for idx in indices_ordenados:
            if autovalores[idx] > tol:
                indices_validos.append(idx)

        r = len(indices_validos)  

        if k == "max" or k > r:
            k = r

        idx_k = indices_validos[:k]

        hatSig = np.sqrt(autovalores[idx_k])

        hatV = S[:, idx_k]
        hatV = normalizarMatriz(hatV, 2)

        B = prodMat(A, hatV)  
        hatU = np.zeros((m, k))

        for j in range(k):
            hatU[:, j] = B[:, j] / hatSig[j]

        return hatU, hatSig, hatV

    else:
        AAt = prodMat(A, At)  

        S, D = diagRH(AAt, tol, K=1000)
        autovalores = np.diag(D).copy()

        indices_ordenados = ordenar_desc(autovalores)

        indices_validos = []
        for idx in indices_ordenados:
            if autovalores[idx] > tol:
                indices_validos.append(idx)

        r = len(indices_validos)

        if k == "max" or k > r:
            k = r

        idx_k = indices_validos[:k]

        hatSig = np.sqrt(autovalores[idx_k])

        hatU = S[:, idx_k]
        hatU = normalizarMatriz(hatU, 2)

        B = prodMat(At, hatU)  
        hatV = np.zeros((n, k))

        for j in range(k):
            hatV[:, j] = B[:, j] / hatSig[j]
            
        return hatU, hatSig, hatV

def svd_reducida_bis(A, k=50, tol=1e-10, K=2000):
    m, n = np.shape(A)

    if m >= n:
        M = prodMat(traspuesta(A), A).astype(float)

        V = np.zeros((n, k))
        Sig = np.zeros(k)

        for i in range(k):
            v, lmbda, _ = metpot2k(M, tol, K)

            sigma = np.sqrt(lmbda)

            if sigma < tol:
                break

            V[:, i] = v
            Sig[i] = sigma

            M = M - lmbda * producto_externo(v, v)

        r = np.sum(Sig > tol)
        V = V[:, :r]
        Sig = Sig[:r]

        U = np.zeros((m, r))
        AV = prodMat(A, V)
        for j in range(r):
            U[:, j] = AV[:, j] / Sig[j]

        return U, Sig, V

    else:
        M = prodMat(A, traspuesta(A)).astype(float)

        U = np.zeros((m, k))
        Sig = np.zeros(k)

        for i in range(k):
            u, lmbda, _ = metpot2k(M, tol, K)
            sigma = np.sqrt(lmbda)

            if sigma < tol:
                break

            U[:, i] = u
            Sig[i] = sigma

            M = M - lmbda * producto_externo(u, u)

        r = np.sum(Sig > tol)
        U = U[:, :r]
        Sig = Sig[:r]

        V = np.zeros((n, r))
        ATU = prodMat(traspuesta(A), U)
        for j in range(r):
            V[:, j] = ATU[:, j] / Sig[j]

        return U, Sig, V

def restri_para_matrices(L, B, inferior=True): 

    f = L.shape[0] 
    cols = B.shape[1] 
    R = np.zeros((f, cols)) 
    
    for j in range(cols): 
        R[:, j] = res_tri(L, B[:, j], inferior) 
    
    return R

########### FUNCIONES PARA EL TP ###########

# Ejercicio 2
""""
imgPts: array de (n, 2) con coordenadas (u, v) en pixeles.
objPts: array de (n, 3) con coordenadas (U, V, W) en el mundo real.
Devuelve Q creada a partir de la ecuación 6
"""
def obtenerQ3d(imgPts, objPts): 
    n = imgPts.shape[0]
    Q = np.zeros((2 * n, 12)) # Creamos una matriz Q llena de ceros de dimensión 2nx12

    # Iteramos sobre los puntos de las imagenes y los objetos
    for i in range(n): # Llenamos la matriz Q en base a los n puntos
        u, v = imgPts[i, 0], imgPts[i, 1]
        U, V, W = objPts[i, 0], objPts[i, 1], objPts[i, 2]

        # Filas pares
        Q[2*i, 0:4] = [U, V, W, 1]
        Q[2*i, 8:12] = [-u * U, -u * V, -u * W, -u]

        #Fila impares
        Q[2*i+1, 4:8] = [U, V, W, 1]
        Q[2*i+1, 8:12] = [-v * U, -v * V, -v * W, -v]

    return Q

""""
Q: matriz creada a partir de la ecuación 6
Devuelve la matriz de proyección M
"""
def resolverDLT3D(Q):

    # k el numero de valores singulares (y vectores) a retener
    # con tolerancia negativa para no borrar valores chicos (que son los que nos importan)
    _,_,V = svd_reducida(Q, k=12, tol=-1) 
    
    m = V[:,-1] # el autovector con el menor autovalor
    M = m.reshape((3,4))

    return M

""""
M: matriz de proyección
Devuelve la matriz de rotación R, el vector de traslación t y la matriz de parámetros intrínsecos K
"""
def devolverMatricesCalibracion3d(M, restriccion=False):
    #Basandonos en la sección 2.1, extraemos R, t y K:
    #M=K[R|t]=[KR|Kt]
    KR = M[:, :3]
    Kt = M[:, 3]

    # Factorización RQ usando QR 
    
    # Creamos la matriz de permutacion P
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    # Calculamos la matriz reversa con P * KR^T * P
    KR_t = traspuesta(KR)
    KR_rev = prodMat(P, prodMat(KR_t, P))
    
    Q, R =  QR_con_GS(KR_rev, tol=1e-12)
    # K = P * R^T * P
    K = prodMat(P, prodMat(traspuesta(R), P))
    # R = P * Q^T * P
    R = prodMat(P, prodMat(traspuesta(Q), P))

    # Normalizar K
    K = K / K[2, 2]

    # los signos de la diagonal deben ser positivos (son distancias)
    T_diag = np.zeros((3, 3))
    for i in range(3):
        if K[i, i] < 0:
            T_diag[i, i] = -1
        else:
            T_diag[i, i] = 1

    K = prodMat(K, T_diag)
    R = prodMat(T_diag, R) 
    
    inv_K = inversa_bis(K)
    t = prodMat(inv_K, Kt)
    
    if restriccion:
        K[0, 1] = 0 # Gamma (sesgo) = 0
        promedio_focal = (K[0, 0] + K[1, 1]) / 2
        K[0, 0] = promedio_focal
        K[1, 1] = promedio_focal

    return K, t, R
    
    # Ejercicio 3   
"""
    3a)
    Por Teorema de la Dimensión: Rg(Q) + dim(Nu(Q)) = 9. Para que el sistema homogéneo Q*h = 0 tenga una 
    solución no nula única salvo por un factor de escala, el núcleo de Q debe tener dimensión 1. Por lo tanto,
    Rg(Q) = 9 - 1 = 8. La matriz Q debe tener rango 8. Lo cual requiere que mínimo n = 4
"""

def obtenerQ2d(imgPts, objPts):
    n = imgPts.shape[0]
    Q = np.zeros((2 * n, 9)) # Creamos una matriz Q llena de ceros de dimensión 2nx9

    for i in range(n): # Llenamos la matriz Q en base a los n puntos
        u, v = imgPts[i, 0], imgPts[i, 1]
        U, V = objPts[i, 0], objPts[i, 1]

        # Filas pares
        Q[2*i, 0:3] = [U, V, 1]
        Q[2*i, 6:9] = [-u * U, -u * V, -u]

        #Fila impares
        Q[2*i+1, 3:6] = [U, V, 1]
        Q[2*i+1, 6:9] = [-v * U, -v * V, -v]

    return Q

def calcular_vij(H, i, j):
    ci = H[:, i]
    cj = H[:, j]
    return np.array([ci[0] * cj[0], 
                     ci[0] * cj[1] + ci[1] * cj[0], 
                     ci[1] * cj[1], 
                     ci[2] * cj[0] + ci[0] * cj[2], 
                     ci[2] * cj[1] + ci[1] * cj[2], 
                     ci[2] * cj[2]])

def armarA(Hlst):
    N = len(Hlst)
    A = np.zeros((2 * N, 6))

    for k in range(N):  
        H = Hlst[k]
        v12 = calcular_vij(H, 0, 1)
        v11 = calcular_vij(H, 0, 0)
        v22 = calcular_vij(H, 1, 1)

        A[2 * k] = v12
        A[2 * k + 1] = v11 - v22

    return A

def calcularMatrizK2D(A):
    _, _, V = svd_reducida(A, k=6, tol=-1)
    b = V[:, -1]

    B = np.array([[b[0], b[1], b[3]], #Re-construimos la matriz simétrica B
                  [b[1], b[2], b[4]], 
                  [b[3], b[4], b[5]]]
                  )

    if B[0, 0] < 0: #Garantizamos que B sea definida positiva
        B = -B

    L = calculaCholesky(B, tol=1e-10) # B = L @ L.T

    Lt = traspuesta(L) 
    K = inversa_bis(Lt) # K = (L.T)^-1 
    K = K / K[2, 2] #Normalizamos K

    return K
    
def obtenerPose2D(Hj, K):

    if Hj[2, 2] < 0: #(t_z > 0) Garantiza que el tablero quede delante de la cámara
        H = -Hj
    else:
        H = Hj

    c1, c2, c3 = H[:, 0], H[:, 1], H[:, 2]

    K_inv = inversa_bis(K)
    k_c1 = prodMat(K_inv, c1)
    k_c2 = prodMat(K_inv, c2)
    k_c3 = prodMat(K_inv, c3)

    lamb = 1 / norma(k_c1, 2) #Factor de escala

    r1 = lamb * k_c1
    r2 = lamb * k_c2
    r3 = prod_vectorial(r1, r2)
    tj = lamb * k_c3

    R_temp = traspuesta(r1, r2, r3)
    R = np.vstack(R_temp)

    U, _, Vt = svd_reducida(R, k=3, tol=-1)
    Rj = prodMat(U, Vt)

    return Rj, tj

############################################ 

# Ejercicio 4 

# 4a
# Pasos usados para calcular la distancia entre la posicion de un jugador y la linea del arco
# 1) Encontrar homografia H^(-1)
# 2) Llevar los pies de un jugador a la cancha: Evaluar H^(-1) [u v 1] = [U V w] para pasar de coordenas de la imagen (u,v) a coordenadas de las cancha (U,V). El resultado está en coordenadas homogeneas. Para recuperar las coordenadas reales de la cancha, dividimos por la tercera componente:
# Ue = U / w, Ve = V / w => Obtenemos los valores del mundo real de los pies de enzo: [Ue, Ve, 1]
# Usamos los pies porque el punto que buscamos debe pertenecer al plano de la cancha
# 3) Calcular la distancia en linea recta del jugador a la linea del arco):  La formula que usamos para calcular la distancia en linea recta sobre el piso es la euclídea, donde calculamos la distancia al punto mas cercano del segmento del arco, que puede ser uno de los palos o un punto entre ellos


"""
    Calcula la homografia H que transforma puntos del mundo real (U, V) a puntos de la imagen (u, v).

    imgPts: puntos de la imagen, de dimension Nx2.
    objPts: puntos del mundo real, de dimension Nx2.

    Devuelve H, una matriz de 3x3.
"""
def calcularHomografia2D(imgPts, objPts):

    Q = obtenerQ2d(imgPts, objPts)

    # Resolvemos Qh = 0 mediante SVD
    _, _, V = svd_reducida(Q, k=9, tol=-1)

    # La solucion es el vector asociado al menor valor singular 
    h = V[:, -1]

    H = h.reshape(3, 3)

    # Normalizamos la homografia 
    H = H / H[2, 2]

    return H

"""
    Transforma un punto de la imagen (u, v) a coordenadas de la cancha (U, V)

    H transforma coordenadas de la cancha a la imagen.
"""
def transformarPuntoCancha(H, imgPts):

    H_inv = inversa_bis(H)

    # Convertimos el punto a coordenadas homogeneas
    p = np.array([imgPts[0], imgPts[1], 1.0])

    # [U V w] = H^(-1) * [u v 1] 
    pCancha = prodMat(H_inv, p)

    # Normalizamos las coordenadas homogeneas: [U V w] / w => [U/w V/w 1]
    U = pCancha[0] / pCancha[2]
    V = pCancha[1] / pCancha[2]

    return U, V

"""
    Calcula la distancia, sobre el piso, desde los pies del jugador hasta el arco

    H: homografía
    imgPts: coordenadas (u, v) de los pies en la imagen.

    Devuelve la distancia en metros
"""
def distanciaPiesArco(H, imgPts):

    # Transformamos los pies desde la imagen hasta la cancha
    # (U, V): posicion de los pies del jugador
    U, V = transformarPuntoCancha(H, imgPts)

    # Cancha: 105 x 68 m
    U_cancha = 105
    # Arco: 7,32m y centrado en la linea de meta
    V_arco = 7.32
    # V_min y V_max son los palos del arco
    V_min = (U_cancha - V_arco) / 2
    V_max = (U_cancha + V_arco) / 2

    # Buscamos el punto del segmento del arco mas cercano a la posicion del jugador
    V_cercano = min(max(V, V_min), V_max)

    # Distancia euclídea en el plano de la cancha
    distancia = np.sqrt( (U - U_cancha)**2 + (V - V_cercano)**2)

    return distancia

# 4b: Debemos ubicar a todos los jugadores y dibujar un circulo de 1 metro de radio sobre el piso
# Hay que definir el circulo en coordenadas de la cancha y proyectar sus puntos a la imagen mediante H.
# 1) Identificar en la imagen los pies de cada jugador: Para cada jugador i tendremos un punto pi = [ui vi 1]
# 2) Aplicar H^(-1) a cada punto para obtener sus coordenadas sobre el campo: para esto utilizamos la funcion transformarPuntoCancha
# 3) Construir un circulo de radio 1 metro alrededor de cada posicion: Una vez que tenemos la posicion de un jugador (Ui,Vi) definimos un circulo mediante U = Ui + cos(ang) y V = Vi + sen(ang) porque (U-Ui)^2 + ( V-Vi)^2 = 1, es decir, la parametrizacion corresponde a un círculo de radio 1 metro
# 4) Proyectar los puntos de esos circulos a la imagen usando H: armamos la funcion proyectarPuntoImagen para ellos
# 5) Dibujar los circulos proyectados sobre la imagen

"""
    Proyecta un punto del mundo real (la cancha en el ejercicio) a coordenadas de imagen
    H [U V 1] = [u v]

    H: homografia del mundo real a la imagen
    objPt: coordenadas (U, V) del punto

    Devuelve las coordenadas (u, v) del punto en la imagen
"""
def proyectarPuntoImagen(H, objPt):

    p = np.array([objPt[0], objPt[1], 1.0])
    p_img = prodMat(H, p)

    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]

    return np.array([u, v])

"""
    Ubica jugadores sobre la cancha y dibuja circulos de radio 1 metro proyectados sobre la imagen

    H: homografia 
    jugPts: lista de puntos (u, v) de los pies de los jugadores

    Devuelve la lista de posiciones de los jugadores en coordenadas de la cancha
"""
def dibujarCirculosJugadores(H, jugPts):

    posCancha = []

    # Angulos para construir los circulos (2 * pi es una vuelta completa)
    angulos = np.arange(0, 2 * math.pi, 0.02)

    for punto in jugPts:

        # Recuperamos la posicion del jugador en la cancha
        U, V = transformarPuntoCancha(H, punto)
        posCancha.append(np.array([U, V]))

        # Construimos los puntos del circulo en coordenadas de cancha
        ptsCirculo = []

        for angulo in angulos:
            U_circulo = U + math.cos(angulo)
            V_circulo = V + math.sin(angulo)

            puntoCancha = np.array([U_circulo, V_circulo])

            # Proyectamos cada punto del circulo a la imagen
            puntoImg = proyectarPuntoImagen(H, puntoCancha)
            ptsCirculo.append(puntoImg)

        ptsCirculo = np.array(ptsCirculo)

        # Dibujamos el circulo sobre la fotografia
        plt.plot(ptsCirculo[:, 0], ptsCirculo[:, 1], linewidth=2)

        # Marca los pies del jugador
        plt.scatter(punto[0], punto[1], s=20)

    plt.show()

    return posCancha

#4c Trazar una recta paralela a la línea de fondo de la cancha que toque el pie del último defensor
#  La recta se define en coordenadas de la cancha y se proyecta a la imagen mediante H
# 1) Identificar puntos de referencia de la cancha y estimar la homografía H para la imagen
# 2) Marcar el pie del ultimo defensor y recuperar su posición (Ud,Vd) en la cancha mediante H^(-1)
# 3) Construir una recta paralela a la línea de fondo que pase por ese punto
# 4) Proyectar la recta sobre la imagen mediante H
# 5) Comparar las posiciones de los otros jugadores con esa linea

"""
    Genera puntos de la recta paralela a la linea de fondo que pasa por U = Ud.

    Ud: Coordenada U del defensor en metros
    Cantidad: Cantidad de puntos que se generan para representar la recta

    Devuelve un arreglo de dimensiones (N, 2), donde cada fila contiene un punto [U, V] de la cancha
"""
def generarRectaCancha(Ud, cantidad=100):

    # Cancha: 105 x 68 m
    Vcancha = 68
    valoresV = np.arange(0, Vcancha, Vcancha / cantidad)

    puntos = []

    for V in valoresV:
        puntos.append([Ud, V])

    return np.array(puntos)

"""
    Proyecta puntos de la cancha a la 
    H: homografia
    ptsCancha: puntos [U, V] de la cancha que se desean proyectar

    Devuelve un arreglo de dimensiones (N, 2) con las coordenadas [u, v] de los  puntos proyectados en la imagen
"""
def proyectarRectaImagen(H, ptsCancha):

    puntosImagen = []

    for punto in ptsCancha:
        puntoImg = proyectarPuntoImagen(H, punto)
        puntosImagen.append(puntoImg)

    return np.array(puntosImagen)

"""
    Compara las posiciones de los jugadores rivales con la recta que pasa por los pies del ultimo defensor, para determinar si estan adelantados segun el sentido de ataque.

    H : homografia

    puntoDefensorImg: Coordenadas [u, v] de los pies del último defensor en la imagen

    puntosRivalesImg: Coordenadas [u, v] de los pies de los jugadores rivales en la imagen

    atacaHaciaUPositivo:
        True si los atacantes avanzan hacia U creciente
        False si avanzan hacia U decreciente
 """
def analizarPosiciones(H, puntoDefensorImg, puntosRivalesImg, atacaHaciaUPositivo=True):
    
    Ud, _ = transformarPuntoCancha(H, puntoDefensorImg)

    resultados = []

    for i, puntoImg in enumerate(puntosRivalesImg):
        Uj, Vj = transformarPuntoCancha(H, puntoImg)

        if atacaHaciaUPositivo:
            adelantado = Uj > Ud
        else:
            adelantado = Uj < Ud

        resultados.append({
            "Jugador": i + 1,
            "U": Uj,
            "V": Vj,
            "U Defendor": Ud,
            "Adelantado?": "Si" if adelantado else "No" 
        })

    return resultados


# Ejercicio 5
#machan

#5a
#Leer el archivo csv
df = pd.read.csv('salguero/world_img_pts.csv', sep=';')
datos = np.asarray(df)

objPts = datos[:, 1:3] #Coordenadas (X, Y) de los 5 puntos, Matirz de (5,2)

H_lista= []

#Usamos Método de Zhang (Ejercicio 3 del TP) para obtener la matriz K

for i in range(4): #Tenemos 4 puntos de vista, armamos una matriz HK para cada uno
    u = 4 + 2 * i
    v = 4 + 2 * i + 1
    imgPts = datos[:, u:v+1] # (uk, vk) coordenadas de los 5 puntos en la imagen k

    Q2d = obtenerQ2d(imgPts, objPts)

    _,_,V = svd_reducida(Q2d, k=9, tol=-1)
    h = V[:,-1] # el vector singular asociado al menor valor singular
    Hk = h.reshape((3,3))

    H_lista.append(Hk)

    A_Zhang = armarA(H_lista)
    K = calcularMatrizK2D(A_Zhang)
#5b



#5c

# Ejercicio 6


