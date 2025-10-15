import pylab, numpy
import perfil   # defines Profile class
import doublespiral_image
import gaus_sub
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
from calcu import disk_base,radial_profile,spiral_arm,gaussian,arcselip,calculations,calcul5,calcul10,calculento
from scipy import interpolate

# Parâmetros base da imagem
lambda0 = 6562.8
ANGIi = 68
broad = 1200
shift = 5  # adicionado manualmente azul ou vermelho do disco
scale = 0.85  # valor escala fluxo
n = 50 # numero de pontos na matriz
xi = 2550  # eixo x da matriz
xf = 2550  # eixo y da matriz

# Matrizes do disco e estruturas
disk = disk_base(-xi, xf, n, 500, 2500)
rad = radial_profile(-xi, xf, n, -0.5, 0.5, 1585)
arm = spiral_arm(-xi, xf, n, 90, 500, -55, 90)
arm2 = spiral_arm(-xi, xf, n, 210, 500, -35, 90)
g1 = gaussian(-xi, xf, n, -1028, -1103, 800, 800, 0, 0)
g2 = gaussian(-xi, xf, n, 1028, -1103, 500, 500, 0, 0)
g3 = gaussian(-xi, xf, n, 2028, 1103, 200, 200, 0, 0)
arcelp = arcselip(-xi, xf, n, 1, 2, numpy.radians(40), 50, 2.67, 6.20, 0.010,numpy.radians(180), 0, -1000, 1200)
# Matriz de cálculo final
disk_final = disk*((arm+arm2))

# Ajuste os valores onde disk_final é zero para 1 (branco)
disk_final[disk_final == 0] = 50


# Configurando o subplot
fig, axs = plt.subplots(2, 2, figsize=(10,7))

# Plot da imagem à direita
axs[0][0].imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0.1, vmax=1, cmap='hot')
axs[0][0].set_xlabel('rg')
axs[0][0].set_ylabel('rg')
axs[0][0].set_facecolor('white')
axs[0][1].set_title('(b)')
axs[0][0].set_title('(a)')
axs[0][0].text(890,1120, r'$\epsilon_{A1}(\phi)$',fontsize=12)
axs[0][0].text(-400,-1520, r'$\epsilon_{A2}(\phi)$',fontsize=12)

lam, flux,vel,points = calcul10(disk_final,lambda0,ANGIi,broad,xf,10)
axs[0][1].plot(lam,flux,lw=2,color='red')
axs[0][1].plot(lam, flux*0, lw=2, color='black')
axs[0][1].set_xlabel('Comprimento de onda (Å)')
axs[0][1].set_ylabel('Fluxo normalizado')





rad = radial_profile(-xi, xf, n, -1.5, 0.8, 200)
arm = spiral_arm(-xi, xf, n, 180, 500, 55, 60)
disk_final = disk*(((arm))+g1+arcelp)
disk_final[disk_final == 0] = 50
# Plot da imagem à direita
axs[1][0].imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0.1, vmax=1, cmap='hot')
axs[1][0].set_xlabel('rg')
axs[1][0].set_ylabel('rg')
axs[1][0].set_facecolor('white')
axs[1][0].set_title('(c)')
axs[1][1].set_title('(d)')
axs[1][0].text(1200,800, r'$\epsilon_A(\phi)$',fontsize=12)
#axs[1].text(700,-1280, r'$\epsilon_{G2}(x,y)$',fontsize=12)
axs[1][0].text(-1500,-1310, r'$\epsilon_{G}(x,y)$',fontsize=12)
axs[1][0].text(-1300,1810, r'$\epsilon_E(x,y)$',color="white",fontsize=12)

lam, flux,vel,points = calcul10(disk_final,lambda0,ANGIi,broad,xf,10)
axs[1][1].plot(lam,flux,lw=2,color='red')
axs[1][1].plot(lam, flux*0, lw=2, color='black')
axs[1][1].set_xlabel('Comprimento de onda (Å)')
axs[1][1].set_ylabel('Fluxo normalizado')


plt.tight_layout()
plt.savefig("discomultiplaspec.jpg",dpi=300)
plt.show()
#plt.close()



import pylab, numpy
import perfil   # defines Profile class
import doublespiral_image
import gaus_sub
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
from calcu import disk_base,radial_profile,spiral_arm,gaussian,arcselip,calculations,calcul5,calcul10,calculento
from scipy import interpolate

# Parâmetros base da imagem
lambda0 = 6562.8
ANGIi = 68
broad = 1200
shift = 5  # adicionado manualmente azul ou vermelho do disco
scale = 0.85  # valor escala fluxo
n = 50 # numero de pontos na matriz
xi = 2550  # eixo x da matriz
xf = 2550  # eixo y da matriz

# Matrizes do disco e estruturas
disk = disk_base(-xi, xf, n, 500, 2500)
rad = radial_profile(-xi, xf, n, -1.5, 2.5, 1585)
arm = spiral_arm(-xi, xf, n, 150, 500, 55, 90)
arm2 = spiral_arm(-xi, xf, n, 140, 500, 35, 90)
g1 = gaussian(-xi, xf, n, -1028, -1103, 800, 800, 0, 0)
g2 = gaussian(-xi, xf, n, 1028, -1103, 500, 500, 0, 0)
g3 = gaussian(-xi, xf, n, 2028, 1103, 200, 200, 0, 0)
arcelp = arcselip(-xi, xf, n, 1, 2, numpy.radians(40), 60, 2.67, 6.20, 0.010,numpy.radians(180), 0, -1000, 1200)
# Matriz de cálculo final
disk_final = disk*(rad*(arm+1.3*g1+arcelp))

# Ajuste os valores onde disk_final é zero para 1 (branco)
disk_final[disk_final == 0] = 50


# Configurando o subplot
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Plot da imagem à direita
axs[0][0].imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0.1, vmax=1, cmap='hot')
axs[0][0].set_xlabel('rg')
axs[0][0].set_ylabel('rg')
axs[0][0].set_facecolor('white')
axs[0][0].set_title('(a)')
axs[0][1].set_title('(b)')
axs[0][0].text(1300,-760, r'$\epsilon_A(\phi)$',fontsize=12)
#axs[0].text(700,-1280, r'$\epsilon_{G2}(x,y)$',fontsize=12)
axs[0][0].text(-1500,-1310, r'$\epsilon_{G}(x,y)$',fontsize=12)
axs[0][0].text(-1300,1810, r'$\epsilon_E(x,y)$',color="white",fontsize=12)


lam, flux,vel,points = calcul10(disk_final,lambda0,ANGIi,broad,xf,10)
axs[0][1].plot(lam,flux,lw=2,color='red')
axs[0][1].plot(lam, flux*0, lw=2, color='black')
axs[0][1].set_xlabel('Comprimento de onda (Å)')
axs[0][1].set_ylabel('Fluxo normalizado')


disk_final = disk*(rad*(arm)+1.3*g1+arcelp)
disk_final[disk_final == 0] = 50
# Plot da imagem à direita
axs[1][0].imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0.1, vmax=1, cmap='hot')
axs[1][0].set_xlabel('rg')
axs[1][0].set_ylabel('rg')
axs[1][0].set_facecolor('white')
axs[1][0].set_title('(c)')
axs[1][1].set_title('(d)')
axs[1][0].text(1300,-760, r'$\epsilon_A(\phi)$',fontsize=12)
#axs[1].text(700,-1280, r'$\epsilon_{G2}(x,y)$',fontsize=12)
axs[1][0].text(-1500,-1310, r'$\epsilon_{G}(x,y)$',fontsize=12)
axs[1][0].text(-1300,1810, r'$\epsilon_E(x,y)$',color="white",fontsize=12)


lam, flux,vel,points = calcul10(disk_final,lambda0,ANGIi,broad,xf,10)
axs[1][1].plot(lam,flux,lw=2,color='red')
axs[1][1].plot(lam, flux*0, lw=2, color='black')
axs[1][1].set_xlabel('Comprimento de onda (Å)')
axs[1][1].set_ylabel('Fluxo normalizado')

plt.tight_layout()
plt.savefig("discomultiplaspec2.jpg",dpi=300)
plt.show()
#plt.close()


"""##################Parametros matriz emissividade#######################################################

center  =6562.8 #central wavelenght
angle   =68 #angulo inclinacao disco em relacao  linha 0=face on
xi1     =500 #>200 raio internto
xi2     =2500#raio externo
xiq     =878 #raio quebra emissividade
amp     =4.1 #ocntraste entre o braco e o brilho subjacente do disco
broad   =1200 #velocity torbulence cel disk--gaussiana espac n rsolvid
q1      =-2.5 #indicide de emissividade raio interna<raioquebra
q2      =3.1#'''''''''''                           ''>raioquebra

######Parametros braço espiral#####
phi00    =0 #angulo inicial do inicio do braco
pitch   =45 #enrolamento do braco
xiw     =127#largura do braco vs brilho
xisp    =xi1 #ponto de inicio do braco


######Parametros gaussiana#####
#   A * np.exp(-(((x - mux) / sigmax)**2 + ((y - muy) / sigmay)**2))
mux=300     ## Media (valor esperado) na dimensao x. Indica o ponto central da Gaussiana na direcao x.
muy=500     # Media (valor esperado) na dimensao y. Indica o ponto central da Gaussiana na direcao y.
sigmax=300     #Desvio padrao na dimensao x. Controla a dispersao dos valores da Gaussiana na direcao x. Quanto maior o valor de "sigmax", mais larga sera a Gaussiana na direcao x.
sigmay=300     #Desvio padrao na dimensao y. Controla a dispersao dos valores da Gaussiana na direcao y. Quanto maior o valor de "sigmay", mais larga sera a Gaussiana na direcao y.
angle_rot=0     #Fator de escala ou amplitude da Gaussiana. Controla a altura do pico da Gaussiana. Quanto maior o valor de "A", mais alto sera o pico da Gaussiana. (nosso mapa valores entre 0-1)
angle_tran=0
#scale=0.0

######Parametros arcelip################
a      =1    #a/b fração de achatamento
b      =2    # a/b fração de achatamento
theta0      =0   # gira a projeção da elipse por um angulo z, manter =0
scale      =100      # aumenta ou diminui ela na matriz é um faot de escala
tangential_stretch      =2      #afasta as ponta da elipse 
radial_stretch      =5.2     #alarga elipse
curvature      =0.004    #curva elipse aumenta a regiao das ponta
rotation_angle      =0 # rotaciona elipse no mesmo ponto da matriz
translation      =0 #translada elipse na matriz
center_x      =1000# posição x da elipse na matriztransalada elipse 
center_y      =1000# posição y da elipse na matriztransalada elipse 

######Parametros base da imagem #########
shift   =5.0 #added manually blue ou red do disco
scale   =0.85#valor escala fluxo
n       =500  #numero de pontos na matriz
xi      =xi2 #eixo x da matrix
xf      =xi2 #eixo y da matriz





#Chamando as função que calculam matriz de emissividade
#disk_base(xi, xf, n, xi1, xi2)
#disk = disk_base(-xi, xf, n, xi1, xi2)

#spiral_arm(xi, xf, n, phi00, xisp, pp, xiww): os braços são independentes pode-se adicionar varios para qualquer posição
#arm = spiral_arm(-xi, xf, n, phi00, xisp, pitch, xiw)

#def radial_profile(xi, xf, n, q1, q2, xiq):
#rad = radial_profile(-xi, xf, n, q1, q2, xiq)

#gaussian(xi, xf, n, mux, muy, sigmax, sigmay,angle_rot,angle_tran)
#g = gaussian(-xi, xf, n, mux, muy, sigmax, sigmay,angle_rot,angle_tran)

#def arcselip(xi, xf, n, a, b, theta0, scale, tangential_stretch, radial_stretch, curvature, rotation_angle,angle_tran, center_x, center_y):
# Gera uma matriz de emissividade em forma de arco elipse com translação
#arcelp = arcselip(-xi, xf, n, 1, 2, numpy.radians(0), 100, 2, 5.2, 0.004, numpy.radians(0),0, -1400, 700)


#função da matriz Disco
#disk_final = disk*(rad*(1+0.5*4.1*(arm))+1.2*arcelp+2*g)


"""


