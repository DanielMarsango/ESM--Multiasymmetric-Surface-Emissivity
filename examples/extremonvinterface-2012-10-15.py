import pylab, numpy
import perfil   # defines Profile class
import doublespiral_image
import gaus_sub
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
from calcu import disk_base,radial_profile,spiral_arm,gaussian,arcselip,calculations,calcul5,calcul10,calculento
from scipy import interpolate

######Parametros base da imagem #########
lambda0	=6562.8
ANGIi   = 68
broad	=1200
shift   =5 #added manually blue ou red do disco
scale   =0.85#valor escala fluxo
n       =50  #numero de pontos na matriz
xi      =2500 #eixo x da matrix
xf      =2500 #eixo y da matriz




###matrizes do disco e estruturas####

#gera matriz base do disco
#disk_base(xi, xf, n, xi1, xi2)
disk = disk_base(-xi, xf,n, 500, 2500)

# Gera uma matriz de emissividade para um disco radial quebrado
#def radial_profile(xi, xf, n, q1, q2, xiq):
#q1 e q2 lei de emissividade radial quebrada q1 valores negativos e Q2 positivos (emissividade cresce e decai)
rad = radial_profile(-xi, xf,n, -2.5, 3.1, 945)

# Gera uma matriz de emissividade para um braço de espiral
#spiral_arm(xi, xf, n, phi00, xisp, pp, xiww): os braços são independentes pode-se adicionar varios para qualquer posição
arm = spiral_arm(-xi, xf,n,83-45, 500, 45, 147)
arm2 = spiral_arm(-xi, xf,n, 77, 400, 20, 60)

# Gera uma matriz de emissividade gaussiana
#gaussian(xi, xf, n, mux, muy, sigmax, sigmay,angle_rot,angle_tran)
g = gaussian(-xi, xf,n, -1028, -73, 500,500,0,90)
g2 = gaussian(-xi, xf,n, -988, -1730, 1000,1000,0,-1)

# Gera uma matriz de emissividade em forma de arco elipse com translação
#def arcselip(xi, xf, n, a, b, theta0, scale, tangential_stretch, radial_stretch, curvature, rotation_angle,trans, center_x, center_y):
arcelp = arcselip(-xi, xf,n, 1, 2, numpy.radians(0), 100, 2, 5.2, 0.004, numpy.radians(0),0, 1000, 1000)


########################################################################################################################################
###############################################################MATRIZ CALCULO############################################################
########################################################################################################################################

disk_final = disk*(rad*(1+0.5*4.1*(arm)+5*arm2+0.2*g)+1.1*g2)




# Configurando o subplot
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot da imagem à direita
axs[0].imshow(disk_final, extent=[-2500, 2500, -2500, 2500], vmin=0, vmax=3, cmap='hot')
axs[0].set_xlabel('rg')
axs[0].set_ylabel('rg')


# Plot dos gráficos wave,flux
#Dados observacinais
specobs='ha_comb_2012-10-15590ecAVG_Pictor_A_5900.fits'

#IFSCUBE model
ifs_model = 'model_ha_comb_2012-10-15590ecAVG_Pictor_A_5900_ifs.txt' 

ifs_wave, ifs_flux = numpy.loadtxt(ifs_model, unpack=True)
obs=perfil.Profile()    # observed spectrum
obs.readfits(specobs)
#obs.readtxt(specobs)

#plot dados observacionais
axs[1].plot(obs.wave,obs.flux, 'b-', lw=1.5, label="Observado")
axs[1].plot(ifs_wave, ifs_flux, lw=1.2, color='magenta', alpha=1.0, label="Ajuste-ifscube")

#Plot model
lam, flux,vel,points = calcul10(disk_final,lambda0,ANGIi,broad,xf,10)
print(lam,flux,vel)
####Scalonando Fluxo
w1 = [6475, 6480]
         
        
i1 = (obs.wave >= w1[0]) & (obs.wave <= w1[1])
i2=(lam >= w1[0]) & (lam <= w1[1])
# Calcular a média e normalizar
mean_fluxobs = np.mean(obs.flux[i1])
mean_flux = np.mean(flux[i2])
scale = mean_fluxobs/mean_flux
#print(scale)

model_wave = lam+shift
model_flux = flux*scale



########################################################chi-Square##############################################################################################################################
#################################################################################################################################################################################################
from scipy.stats import chisquare
from scipy.interpolate import interp1d

# Definir os intervalos de interesse
intervalo1 = (obs.wave >= 6400) & (obs.wave <= 6507)
intervalo2 = (obs.wave >= 6617) & (obs.wave <= 6700)

# Criar função de interpolação para os dados do modelo
interp_model_flux = interp1d(model_wave, model_flux, kind='linear', fill_value='extrapolate')

# Avaliar as funções interpoladas nos intervalos de observação
model_flux_intervalo1 = interp_model_flux(obs.wave[intervalo1])
model_flux_intervalo2 = interp_model_flux(obs.wave[intervalo2])

# Concatenar os intervalos para formar o intervalo combinado
intervalo_combinado = np.concatenate([obs.wave[intervalo1], obs.wave[intervalo2]])
fluxo_combinado_obs = np.concatenate([obs.flux[intervalo1], obs.flux[intervalo2]])
fluxo_combinado_modelo = np.concatenate([model_flux_intervalo1, model_flux_intervalo2])

# Plotar o intervalo combinado
#plt.plot(intervalo_combinado, fluxo_combinado_obs, 'b-', lw=3.5, label="Observado - Intervalo Combinado")
#plt.plot(intervalo_combinado, fluxo_combinado_modelo, "r-", lw=3.5, label="Modelo ajustado - Intervalo Combinado")
# Plotar Intervalo 1
axs[1].plot(obs.wave[intervalo1], obs.flux[intervalo1], 'b-', lw=3.5, label="Observado - Intervalo 1")
axs[1].plot(obs.wave[intervalo1], model_flux_intervalo1, "r-", lw=3.5, label="Modelo ajustado - Intervalo 1")

# Plotar Intervalo 2
axs[1].plot(obs.wave[intervalo2], obs.flux[intervalo2], 'b-', lw=3.5, label="Observado - Intervalo 2")
axs[1].plot(obs.wave[intervalo2], model_flux_intervalo2, "r-", lw=3.5, label="Modelo ajustado - Intervalo 2")
# Normalizar as frequências observadas e esperadas
observed_normalized_combinado = fluxo_combinado_obs / np.mean(fluxo_combinado_obs)
expected_normalized_combinado = fluxo_combinado_modelo / np.mean(fluxo_combinado_modelo)

# Calcular chi-square para o intervalo combinado
chi2_combinado, p_value_combinado = chisquare(f_obs=observed_normalized_combinado, f_exp=expected_normalized_combinado)

# Examinar os resultados
print(f'Chi-square para o intervalo combinado: {chi2_combinado}, p-value: {p_value_combinado}')



#################Plots##########################################
#################################################################
axs[1].plot(model_wave, model_flux, "r--", label="Modelo")

#subtrair do original para verificar se reproduziu toda linha larga
interp_model_flux = interpolate.interp1d(model_wave, model_flux, kind='linear', fill_value='extrapolate')
# Interpolate model_flux based on obs.wave
model_flux_interp = interp_model_flux(obs.wave)
# Subtract model_flux_interp from obs.flux
subtracted_flux = obs.flux - model_flux_interp
axs[1].plot(obs.wave, subtracted_flux,'g-',lw=1.15, label="Residual")

#linha em zero
zero = numpy.zeros(numpy.size(obs.wave))
axs[1].plot(obs.wave, zero, 'k--')


###outras funções para calculo da emissividade
#lam, flux = calcul5(disk_final)
#axs[1].plot(lam, flux, "o-", label="Novo-passo5")

#lam, flux = calculento(disk_final)
#axs[1].plot(lam,flux,"k--",label="lento")


axs[1].legend()
axs[1].set_xlabel('Comprimento de Onda (Â)')
axs[1].set_ylabel('Fluxo') 

plt.tight_layout()
plt.legend()
plt.savefig(specobs[0:18]+".jpg")
plt.show()
#plt.close()
np.savetxt('novodisco'+ifs_model,np.c_[model_wave, model_flux])
np.savetxt('disk_final_'+ifs_model, disk_final)



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

