import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt
import numpy, math
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from numba import jit
from multiprocessing import Pool, cpu_count

@jit(forceobj=True)
def disk_base(xi, xf, n, xi1, xi2):
    # Cria uma malha de coordenadas X e Y para os cálculos
    #Xi, Xf  e N tamanho da matriz e numeros de pontos
    #Xi1 e Xi2 raio interno e externo do Disco
    x = numpy.linspace(xi, xf, 10 * n)
    y = x.copy()
    X, Y = numpy.meshgrid(x, y)

    # Matriz de emissividade inicializada com zeros
    bright = numpy.zeros_like(X)

    # Calcula o raio a partir das coordenadas X e Y
    r = numpy.sqrt(X**2 + Y**2)

    # Define uma região com emissividade não nula
    I = numpy.where((r >= xi1) & (r <= xi2))
    bright[I] = 1

    return bright.T
    
    
@jit(forceobj=True)
def radial_profile(xi, xf, n, q1, q2, xiq):
    # Cria uma malha de coordenadas X e Y para os cálculos
    #Xi, Xf  e N tamanho da matriz e numeros de pontos
    #q1 e q2 lei de emissividade radial quebrada q1 valores negativos e Q2 positivos (emissividade cresce e decai)
    x = numpy.linspace(xi, xf, 10 * n)
    y = x.copy()
    X, Y = numpy.meshgrid(x, y)

    # Matriz de emissividade inicializada com zeros
    bright = numpy.zeros_like(X)

    # Calcula o raio e o ângulo polar
    r = numpy.sqrt(X**2 + Y**2)
    theta = numpy.arctan2(Y, X)

    # Calcula a emissividade baseada nos parâmetros fornecidos
    norm1 = xiq**(-q1)
    norm2 = xiq**(-q2)
    i = numpy.where(r <= xiq)
    bright[i] = r[i]**(-q1)
    j = numpy.where(r > xiq)
    bright[j] = (r[j]**(-q2)) * (norm1 / norm2)
    bright = bright / bright.max()

    return bright.T
    
    
@jit(forceobj=True)
def spiral_arm(xi, xf, n, phi00, xisp, pp, xiww):
    #Xi, Xf  e N tamanho da matriz e numeros de pontos
    #phi00 angulo inicial do braço
    #xisp posição inicial do braço geralmente xi1
    #pp pitch angle
    #xiw alargamento do braço
    
    # Converte ângulos para radianos
    phi0 = math.radians(phi00)% (2 * np.pi)
    p = math.radians(pp)% (2 * np.pi)
    xiw = math.radians(xiww)% (2 * np.pi)

    # Cria uma malha de coordenadas X e Y para os cálculos
    x = numpy.linspace(xi, xf, 10 * n)
    y = x.copy()
    X, Y = numpy.meshgrid(x, y)

    # Matriz de emissividade inicializada com zeros
    bright = numpy.zeros_like(X)

    # Calcula o raio e o ângulo polar
    r = numpy.sqrt(X**2 + Y**2)
    theta = numpy.arctan2(Y, X)

    # Calcula os valores de emissividade para um braço de espiral
    l = numpy.where(r >= xisp)
    
    #psi0 = phi0 + numpy.log10(r) / math.tan(p) esse psi zera em 360
    # expressão está calculando psi0 a partir de phi0, r e p, envolvendo logaritmos naturais e funções trigonométricas, e garantindo que o resultado esteja dentro do intervalo de 0 a 2 * pi
    psi0 = (phi0 + numpy.log(r) / np.tan(p)) % (2 * np.pi)
    dphi = abs(theta - psi0)
    bright[l] = 1/2.0 * (
        numpy.exp(-4.0 * math.log(2.0) / (xiw**2) * dphi[l]**2) +
        numpy.exp(-4.0 * math.log(2.0) / (xiw**2) * (2.0 * math.pi - dphi[l])**2))

    bright = bright / bright.max()

    return bright.T

# Função para gerar matriz de emissividade gaussiana
@jit(forceobj=True)
def gaussian(xi, xf, n, mux, muy, sigmax, sigmay, angle_rot, angle_tran):
	
    #cria uma função gaussiana que tem parametros de rotação e translação sobre o disco)
    #Xi, Xf  e N tamanho da matriz e numeros de pontos
    #mux e muy posição x,y da gaussiana
    #sigma x,y alargamento para cada coordenada
    #angle angulo de rotação da gaussiana
    #angle_trans angulo transtação gaussiana no disco
    # Xi, Xf  e N tamanho da matriz e números de pontos
    muxx=mux
    mux=-muy
    muy=-muxx
    x = numpy.linspace(xi, xf, n * 10)
    y = x.copy()
    X, Y = numpy.meshgrid(x, y)
    
    # Matriz de emissividade inicializada com zeros
    bright = numpy.zeros_like(X)
    
    # Define parâmetro de rotação da Gaussiana
    # Angle é o ângulo de rotação da gaussiana
    rotation = numpy.deg2rad(angle_rot)
    
    # Calcula os valores da gaussiana
    # Mux e muy são as posições x e y da gaussiana
    center_x = X * numpy.cos(rotation) - Y * numpy.sin(rotation)
    center_y = X * numpy.sin(rotation) + Y * numpy.cos(rotation)
    xp = mux * numpy.cos(rotation) + muy * numpy.sin(rotation)
    yp = mux * numpy.sin(rotation) - muy * numpy.cos(rotation)
    bright = 1 * numpy.exp(-(((center_x - xp) / sigmax)**2 + ((center_y - yp) / sigmay)**2))
    
    # Converte o ângulo de translação de graus para radianos
    angle_rad = math.radians(angle_tran)
    
    # Calcula as coordenadas rotacionadas na matriz de rotação
    X_rot = X * numpy.cos(angle_rad) - Y * numpy.sin(angle_rad)
    Y_rot = X * numpy.sin(angle_rad) + Y * numpy.cos(angle_rad)
    
    # Calcula as coordenadas interpolação para mapear de volta para as coordenadas originais
    X_interp = (X_rot - xi) / (xf - xi) * (n * 10 - 1)
    Y_interp = (Y_rot - xi) / (xf - xi) * (n * 10 - 1)
    
    # Mapeia as coordenadas interpolação de volta para as coordenadas originais da matriz
    rotated_bright = map_coordinates(bright, [Y_interp, X_interp], mode='constant', cval=0)
    bright = rotated_bright
    bright = bright / bright.max()
 
    return bright.T



# Função para gerar matriz de emissividade para um arco elíptico gravitacional
   
@jit(forceobj=True)
def arcselip(xi, xf, n, a, b, theta0, scale, tangential_stretch, radial_stretch, curvature, rotation_angle,angle_tran, center_x, center_y):
    x = numpy.linspace(xi, xf, 10 * n)
    y = x.copy()
    X, Y = numpy.meshgrid(x, y)
    center_xx=center_x
    center_x=-center_y
    center_y=center_xx	
    # Matriz de emissividade inicializada com zeros
    bright = numpy.zeros_like(X)
    x_stretched = (X - center_x) / radial_stretch
    y_stretched = (Y - center_y) / (tangential_stretch * numpy.cos(theta0) + numpy.sin(theta0))
    
    x_rotated = x_stretched * numpy.cos(rotation_angle) - y_stretched * numpy.sin(rotation_angle)
    y_rotated = x_stretched * numpy.sin(rotation_angle) + y_stretched * numpy.cos(rotation_angle)
    
    x_curved = x_rotated
    y_curved = y_rotated + curvature * (x_rotated**2)
    
    ellipse_equation = ((x_curved * numpy.cos(theta0) + y_curved * numpy.sin(theta0)) / (a * scale))**2 + ((y_curved * numpy.cos(theta0) - x_curved * numpy.sin(theta0)) / (b * scale))**2
    bright = numpy.exp(-ellipse_equation)
    # Converte o ângulo de translação de graus para radianos
    angle_rad = math.radians(angle_tran)
    
    # Calcula as coordenadas rotacionadas na matriz de rotação
    X_rot = X * numpy.cos(angle_rad) - Y * numpy.sin(angle_rad)
    Y_rot = X * numpy.sin(angle_rad) + Y * numpy.cos(angle_rad)
    
    # Calcula as coordenadas interpolação para mapear de volta para as coordenadas originais
    X_interp = (X_rot - xi) / (xf - xi) * (n * 10 - 1)
    Y_interp = (Y_rot - xi) / (xf - xi) * (n * 10 - 1)
    
    # Mapeia as coordenadas interpolação de volta para as coordenadas originais da matriz
    rotated_bright = map_coordinates(bright, [Y_interp, X_interp], mode='constant', cval=0)
    bright = rotated_bright
    bright = bright / bright.max()
    
    return bright.T
    
@jit(forceobj=True)
def calculations(POINTS,NARMS,OLAMBDA, ANGIi, XI1, XI2, XIq, BROADd, PHI00, Q1, Q2, AMP, PITCHh, XIW, XISP):
    pi = 3.141592
    nstep = 100
    maxstep = 100
    maxwave = POINTS
    maxarms = NARMS
    
    EX = np.empty(maxwave)
    CLAM = np.empty(maxwave)
    VEL = np.empty(maxwave)
    PHIP = np.empty(maxstep)
    SINCOS = np.empty(maxstep)
    SINSIN = np.empty(maxstep)
    
    PHI0 = PHI00 / 57.29578
    ANGI = ANGIi / 57.29578
    SINI = np.sin(ANGI)
    COSI = np.cos(ANGI)
    TANI = np.tan(ANGI)
    BROAD = BROADd / 3.E5
    PITCH = PITCHh / 57.29578
    
    SIG = (pi / 180.) * XIW / np.sqrt(8. * np.log(2.))
    TANPITCH = np.tan(PITCH)
    
    XIDEL = np.log10(XI2 / XI1) / nstep / 2.
    XIDEL = 10. ** XIDEL
    XIDIFF = (XIDEL - 1. / XIDEL)
    PHISTEP = 2. * 3.14159 / nstep
    
    for i in range(nstep):
        PHIP[i] = 0.5 * PHISTEP * (2 * i - 1)
        SINCOS[i] = SINI * np.cos(PHIP[i])
        SINSIN[i] = SINI * np.sin(PHIP[i])
    
    NORM1 = XIq ** (1 - Q1)
    NORM2 = XIq ** (1 - Q2)
    
    for k in tqdm(range(maxwave), desc="Calculating", leave=True):
        EX[k] = 0.0705 - 0.0005 * k
        CLAM[k] = 0.
        
        for j in range(nstep):
            XI = XI1 * XIDEL ** (2 * j - 1)
            XISTEP = XI * XIDIFF
            ALPHA = np.sqrt(1. - (3. / XI))
            BETA = np.sqrt(1. - (2. / XI))
            
            if XI <= XIq:
                XIPOW = XI ** (1 - Q1)
            else:
                XIPOW = XI ** (1 - Q2) * (NORM1 / NORM2)
            
            PSI0 = PHI0 + np.log10(XI / XI2) / TANPITCH
            
            for i in range(nstep):
                ARMS = 0.
                
                if XI >= XISP:
                    for n in range(1, maxarms + 1):
                        PSI = PSI0 + 2. * pi * (n - 1) / maxarms
                        DPSI = np.abs(PHIP[i] - PSI)
                        ARMS = ARMS + np.exp(-DPSI * DPSI / 2. / SIG / SIG)
                        DPSI = 2. * pi - np.abs(PHIP[i] - PSI)
                        ARMS = ARMS + np.exp(-DPSI * DPSI / 2. / SIG / SIG)
                
                DOPPLER = 1. + (SINSIN[i] / np.sqrt(XI))
                DOPPLER = ALPHA / DOPPLER
                EXPON = (1. + EX[k] - DOPPLER) / DOPPLER / BROAD
                EXPON = EXPON * EXPON / 2.
                ARG = DOPPLER ** 3 * XIPOW * (1. + 0.5 * AMP * ARMS) * \
                      (1 + ((1 - SINCOS[i]) / (1 + SINCOS[i])) / XI) * np.exp(-EXPON)
                ELEMENT = ARG * XISTEP * PHISTEP
                CLAM[k] = CLAM[k] + ELEMENT
        
        VEL[k] = -3.E5 * EX[k]
        EX[k] = OLAMBDA / (1. + EX[k])
    CLAM=CLAM/CLAM.max()    
    return EX, CLAM, VEL
    

@jit(forceobj=True)  
def calculento(disk, lam0,ang,broad):
    # Obter as dimensões da matriz do disco
    rows, cols = disk.shape

    xc, yc = cols // 2, rows // 2

    XX, YY = np.meshgrid(np.arange(0, rows, 1), np.arange(0, cols, 1))

    RR = np.sqrt((XX - xc)**2 + (yc - YY)**2)

    THETA = np.arctan2(yc - YY, XX - xc)

    THETA = np.where(THETA < 0, THETA + 2 * np.pi, THETA)
    
    
    #plt.imshow(THETA)
    #plt.show()
    # Inicializar matrizes para R e θ
    R = np.zeros_like(disk)
    theta = np.zeros_like(disk)

    # Criar matrizes para armazenar os vetores de onda, fluxo e velocidade
    wave = np.zeros_like(disk)
    flux = np.zeros_like(disk)
    velocidade = np.zeros_like(disk)

    # Parâmetros
    c = 3.0e5  # Velocidade da luz em km/s

    # Criando vetor wave
    num_points = 300
    interval_size = 0.1 / (num_points)
    z_vec = np.arange(-0.05, 0.05, interval_size)
    f_line = np.zeros(np.size(z_vec))
    wave = 6563 / (1 + z_vec)

    print(np.size(z_vec))

    #calculando vel
    velocidade = -c * z_vec
    #print("velocidade",velocidade)	
    
    
    
    # Contagem de pontos com fluxo positivo
    co = np.count_nonzero(disk > 0)
    progress_bar = tqdm(total=co, desc="Conversão", unit="ponto")

    # Loop através de cada ponto da matriz do disco
    for i in range(rows):
        for j in range(cols):
            flux = disk[i, j]
            if flux > 0:
                # Obter as coordenadas x e y do ponto
                x = j - cols // 2
                y = rows // 2 - i

                # Calcular o raio R e o ângulo polar θ
                R = np.sqrt(x**2 + y**2)

                # Ajustar o ângulo para o intervalo de 0 a 2π
                theta = np.arctan2(x, y)
                theta = np.where(theta < 0, theta + 2 * np.pi, theta)
		
                ALPHA = np.sqrt(1. - (3. / R))
                BETA = np.sqrt(1. - (2. / R))

                BROAD = broad / c

                SINCOS = np.sin(ang * np.pi / 180) * np.cos(theta)
                SINSIN = np.sin(ang * np.pi / 180) * np.sin(theta)

                DOPPLER = 1. + (SINSIN / np.sqrt(R))
                DOPPLER = ALPHA / DOPPLER
                EXPON = (1. + z_vec - DOPPLER) / DOPPLER / BROAD
                EXPON = EXPON * EXPON / 2.

                ARG = DOPPLER ** 3 * flux * (1 + ((1 - SINCOS) / (1 + SINCOS)) / R) * np.exp(-EXPON)
                f_line = f_line + ARG

                # Atualizar o progresso
                progress_bar.update(1)

    # Fechar a barra de progresso
    progress_bar.close()
    f_line=f_line/f_line.max()
    return wave, f_line,velocidade



@jit(forceobj=True)
def calcul5(disk, lam0,ang,broad,xif, nmatrix):
    # Obter as dimensões da matriz do disco
    rows, cols = disk.shape
    
  
    

    xc, yc = cols // 2, rows // 2

    XX, YY = np.meshgrid(np.arange(0, rows, 10), np.arange(0, cols, 10))

    RR = np.sqrt((XX - xc) ** 2 + (yc - YY) ** 2)

    THETA = np.arctan2(yc - YY, XX - xc)

    THETA = np.where(THETA < 0, THETA + 2 * np.pi, THETA)

    # Inicializar matrizes para R e θ
    R = np.zeros_like(disk)
    theta = np.zeros_like(disk)

    # Criar matrizes para armazenar os vetores de onda, fluxo e velocidade
    wave = np.zeros_like(disk)
    flux = np.zeros_like(disk)
    velocidade = np.zeros_like(disk)

    # Parâmetros
    c = 3.0e5  # Velocidade da luz em km/s

    # Criando vetor wave
    num_points = 300
    interval_size = 0.1 / (num_points)
    z_vec = np.arange(-0.06, 0.06, interval_size)
    f_line = np.zeros(np.size(z_vec))
    wave = lam0 / (1 + z_vec)

    print(np.size(z_vec))

    #calculando vel
    velocidade = -c * z_vec
    #print("velocidade",velocidade)	

    #Calculo do espectro


    # Contagem de pontos com fluxo positivo
    co = np.count_nonzero(disk[::nmatrix, ::nmatrix] > 0)
    
    progress_bar = tqdm(total=co, desc="Conversão", unit="ponto")

    point_coordinates = []
    # Loop através de cada ponto da matriz do disco
    
    
    nx = np.size(range(0, rows, nmatrix))
    ny = np.size(range(0, cols, nmatrix))
    for i in range(0, rows, nmatrix):
        for j in range(0, cols, nmatrix):
            flux = disk[i, j]
            if flux > 0:
                

                dx = 2*xif/nx
                dy = 2*xif/ny
                # Obter as coordenadas x e y do ponto
                x = (j - cols // 2)*2*xif/cols
                y = (rows // 2 - i)*2*xif/rows
                
                #x1 = (j + 1 - cols // 2)*2*xi2/cols
                #y = (rows // 2 - i)*2*xi2/rows
                point_coordinates.append((i, j))
                # Calcular o raio R e o ângulo polar θ
                R = np.sqrt(x ** 2 + y ** 2)

                # Ajustar o ângulo para o intervalo de 0 a 2π
                theta = np.arctan2(x, y)
                theta = np.where(theta < 0, theta + 2 * np.pi, theta)

                ALPHA = np.sqrt(1. - (3. / R))
                BETA = np.sqrt(1. - (2. / R))

                
                BROAD = broad / c

                SINCOS = np.sin(ang * np.pi / 180) * np.cos(theta)
                SINSIN = np.sin(ang * np.pi / 180) * np.sin(theta)

                DOPPLER = 1. + (SINSIN / np.sqrt(R))
                DOPPLER = ALPHA / DOPPLER
                EXPON = (1. + z_vec - DOPPLER) / DOPPLER / BROAD
                EXPON = EXPON * EXPON / 2.

                ARG = DOPPLER ** 3 * flux * (1 + ((1 - SINCOS) / (1 + SINCOS)) / R) * np.exp(-EXPON)*dx*dy
                f_line = f_line + ARG

                # Atualizar o progresso
                progress_bar.update(1)

    
    # Converta as coordenadas para um array numpy
    point_coordinates = np.array(point_coordinates)

    # Exiba a matriz usando imshow
    #plt.imshow(disk, cmap='hot')

    # Adicione o contorno em torno dos pontos
    # Plote os pontos nas coordenadas especificadas
    #plt.scatter(point_coordinates[:, 1], point_coordinates[:, 0], color='green', alpha=0.5, marker='s',label='Coord-5')
    

    # Fechar a barra de progresso
    progress_bar.close()
    f_line = f_line / f_line.max()
    return wave, f_line,velocidade,point_coordinates
    
    
@jit(forceobj=True)
def calcul10(disk, lam0,ang,broad,xif,nmatrix):
    # Obter as dimensões da matriz do disco
    rows, cols = disk.shape
   
     

    xc, yc = cols // 2, rows // 2

    XX, YY = np.meshgrid(np.arange(0, rows, 10), np.arange(0, cols, 10))

    RR = np.sqrt((XX - xc) ** 2 + (yc - YY) ** 2)

    THETA = np.arctan2(yc - YY, XX - xc)

    THETA = np.where(THETA < 0, THETA + 2 * np.pi, THETA)

    # Inicializar matrizes para R e θ
    R = np.zeros_like(disk)
    theta = np.zeros_like(disk)

    # Criar matrizes para armazenar os vetores de onda, fluxo e velocidade
    wave = np.zeros_like(disk)
    flux = np.zeros_like(disk)
    velocidade = np.zeros_like(disk)

    # Parâmetros
    c = 3.0e5  # Velocidade da luz em km/s

    # Criando vetor wave
    num_points = 300
    interval_size = 0.1 / (num_points)
    z_vec = np.arange(-0.06, 0.06, interval_size)
    f_line = np.zeros(np.size(z_vec))
    wave = lam0 / (1 + z_vec)

    print(np.size(z_vec))

    #calculando vel
    velocidade = -c * z_vec
    #print("velocidade",velocidade)	

   #Calculo do espectro


    # Contagem de pontos com fluxo positivo
    co = np.count_nonzero(disk[::nmatrix, ::nmatrix] > 0)
    
    progress_bar = tqdm(total=co, desc="Conversão", unit="ponto")

    point_coordinates = []
    # Loop através de cada ponto da matriz do disco
    
    
    nx = np.size(range(0, rows, nmatrix))
    ny = np.size(range(0, cols, nmatrix))
    for i in range(0, rows, nmatrix):
        for j in range(0, cols, nmatrix):
            flux = disk[i, j]
            if 0 < flux < 50:
                

                dx = 2*xif/nx
                dy = 2*xif/ny
                # Obter as coordenadas x e y do ponto
                x = (j - cols // 2)*2*xif/cols
                y = (rows // 2 - i)*2*xif/rows
                
                point_coordinates.append((i, j))
                # Calcular o raio R e o ângulo polar θ
                R = np.sqrt(x ** 2 + y ** 2)

                # Ajustar o ângulo para o intervalo de 0 a 2π
                theta = np.arctan2(x, y)
                theta = np.where(theta < 0, theta + 2 * np.pi, theta)

                ALPHA = np.sqrt(1. - (3. / R))
                BETA = np.sqrt(1. - (2. / R))

                BROAD = broad / c

                SINCOS = np.sin(ang * np.pi / 180) * np.cos(theta)
                SINSIN = np.sin(ang * np.pi / 180) * np.sin(theta)

                DOPPLER = 1. + (SINSIN / np.sqrt(R))
                DOPPLER = ALPHA / DOPPLER
                EXPON = (1. + z_vec - DOPPLER) / DOPPLER / BROAD
                EXPON = EXPON * EXPON / 2.

                ARG = DOPPLER ** 3 * flux * (1 + ((1 - SINCOS) / (1 + SINCOS)) / R) * np.exp(-EXPON)*dx*dy
                f_line = f_line + ARG

                # Atualizar o progresso
                progress_bar.update(1)
    # Converta as coordenadas para um array numpy
    point_coordinates = np.array(point_coordinates)

    # Exiba a matriz usando imshow
    #plt.imshow(disk, cmap='hot')

    # Adicione o contorno em torno dos pontos
    # Plote os pontos nas coordenadas especificadas
    #plt.scatter(point_coordinates[:, 1], point_coordinates[:, 0], color='blue', label='Coord-10')
    


    # Fechar a barra de progresso
    progress_bar.close()
    f_line = f_line / f_line.max()
    return wave, f_line,velocidade,point_coordinates    
# Example usage
if __name__ == '__main__':


    #gera matriz base do disco
    #disk_base(xi, xf, n, xi1, xi2)
    disk = disk_base(-2500, 2500, 500, 500, 2500)


    # Gera uma matriz de emissividade para um braço de espiral
    #spiral_arm(xi, xf, n, phi00, xisp, pp, xiww): os braços são independentes pode-se adicionar varios para qualquer posição
    arm = spiral_arm(-2500, 2500, 500,80-45, 500, 45, 127)
    #arm2 = spiral_arm(-2500, 2500, 500, 30, 0, 60, 60)
	
	
	
	
    # Gera uma matriz de emissividade para um disco radial quebrado
    #def radial_profile(xi, xf, n, q1, q2, xiq):
    #q1 e q2 lei de emissividade radial quebrada q1 valores negativos e Q2 positivos (emissividade cresce e decai)
    rad = radial_profile(-2500, 2500, 500, -2.5, 3.1, 878)
	
	
	
    # Gera uma matriz de emissividade gaussiana
    #gaussian(xi, xf, n, mux, muy, sigmax, sigmay,angle_rot,angle_tran)
    g = gaussian(-2500, 2500, 500, -300, -200, 300,300,0,0)
	
    #def arcselip(xi, xf, n, a, b, theta0, scale, tangential_stretch, radial_stretch, curvature, rotation_angle,trans, center_x, center_y):
    # Gera uma matriz de emissividade em forma de arco elipse com translação
    arcelp = arcselip(-2500, 2500, 500, 1, 2, numpy.radians(0), 100, 2, 5.2, 0.004, numpy.radians(0),0, 1000, 1000)


    disk_final = disk*(rad*(1+0.5*4.1*(arm)))
    #disk_final = disk*(rad)
    #disk_final=disk_final/disk_final.max()
    #disk_final = disk*(g+arm)
    #disk_final=arcelp*disk
    
    #plot disk emissivity
    #plt.imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0, vmax=3,cmap='hot')
    #plt.show()
    
    
    # Configurando o subplot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    

    # Plot da imagem à direita
    axs[0].imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0, vmax=3, cmap='hot')
    axs[0].set_xlabel('rg')
    axs[0].set_ylabel('rg')


    
    # Plot dos gráficos à esquerda

    
    # Plot do gráfico adicional (se necessário) à direita
    # axs[2].plot(lam, flux, "k--", label="lento")
    # axs[2].legend()
    # axs[2].set_xlabel('Lambda')
    # axs[2].set_ylabel('Fluxo')

    lam, flux,vel = calcul10(disk_final,6562.8,68,1200,2500,10)
    #print(vel)
    #print(lam)
    #print(flux)
    
    #plt.plot(lam,flux,"r--",label="Novo-passo10")
    axs[1].plot(lam, flux, "r--", label="Novo-passo10")

        
    lam, flux,vel = calcul5(disk_final,6562.8,68,1200,2500,5)
    #print(lam)
    #print(flux)
    #print("novo vel",vel)
    #plt.plot(lam,flux,"o-",label="Novo-passo5")
    axs[1].plot(lam, flux, "o-", label="Novo-passo5")

    #lam, flux,vel = calculento(disk_final,6562.8,68,1200)
    #print(lam)
    #print(flux)
    #axs[1].plot(lam,flux,"k--",label="lento")


    #### modelo antigo comparação##############

    POINTS=300
    NARMS=1
    OLAMBDA = 6562.8
    ANGIi = 68
    XI1 = 500
    XI2 = 2500
    XIq = 878
    BROADd = 1200
    PHI00 = 80
    Q1 = -2.5
    Q2 = 3.1
    AMP = 4.1
    PITCHh = 45
    XIW = 127
    XISP = XI1

	
    lam, flux, vel = calculations(POINTS,NARMS,OLAMBDA, ANGIi, XI1, XI2, XIq, BROADd, PHI00, Q1, Q2, AMP, PITCHh, XIW, XISP)
    #print(lam)
    #print(flux)
    #print("velho vel",vel)
    axs[1].plot(lam, flux, label="antigo")
    axs[1].legend()
    axs[1].set_xlabel('Lambda')
    axs[1].set_ylabel('Fluxo') 
        
    plt.tight_layout()
    plt.legend()
    plt.show()

    
    
