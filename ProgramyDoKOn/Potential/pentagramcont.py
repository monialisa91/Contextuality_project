import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import time
import matplotlib.pyplot as plt

def mulvec(a,b):
	return np.einsum('ik,k->i',a,b)

def mulmul(a,b):
	return np.einsum ('ik,kj->ij',a,b)

def mulmul3(a,b,c):
	return np.einsum('ik,kj,jm->im',a,b,c)

def vectorLength(vector):
	suma = 0
	for i in range(0, len(vector)):
		suma += vector[i] ** 2
	return np.sqrt(suma)

# Inner Product for Complex Vectors (numpy provides only for the real ones)

def dotpr(v1,v2):
	v2 = np.conjugate(v2)
	return np.inner(v1,v2)

def outerpr(v1,v2):
	v1 = np.conjugate(v1)
	return np.outer(v1,v2)


#                                           PARAMETERS

# Mixing angles (sth- sinus 2*theta, cth- cosinus 2*theta)

STH13 = np.sqrt(0.021)
STH23 = np.sqrt(0.5)
STH12 = np.sqrt(0.307)

CTH13 = np.sqrt(1 - STH13 ** 2)
CTH23 = np.sqrt(1 - STH23 ** 2)
CTH12 = np.sqrt(1 - STH12 ** 2)


print("STH13:{}".format(STH13))
print("STH23:{}".format(STH23))
print("STH12:{}".format(STH12))

print("CTH13:{}".format(CTH13))
print("CTH23:{}".format(CTH23))
print("CTH12:{}".format(CTH12))


# The mass squares

m21 = 7.53 * 10
m32 = 2.52 * 10 ** (3)
m31 = m21 + m32

# The remaining parameters

En = 10 ** 10 # energy of the neutrino
VCC = 0.1 # charge-current potential
h = 0.01
delta = 0

#                                           MIXING MATRIX


Mixing = [[0 for i in range(0,3)] for j in range(0,3)] # empty 3-dimensional matrix
delta = 0

Mixing[0][0] = CTH12 * CTH13
Mixing[0][1] = STH12 * CTH13
Mixing[0][2] = STH13 * np.exp(-1j * delta)

Mixing[1][0] =-STH12 * CTH23 - CTH12 * STH23 * STH13 * np.exp(-1j * delta)
Mixing[1][1] = CTH12 * CTH23 - STH12 * STH23 * STH13 * np.exp(-1j * delta)
Mixing[1][2] = STH23 * CTH13

Mixing[2][0] = STH12 * STH23 - CTH12 * CTH23 * STH13 * np.exp(-1j * delta)
Mixing[2][1] = -CTH12 * STH23 - STH12 * CTH23 * STH13 * np.exp(-1j * delta)
Mixing[2][2] = CTH13 * CTH23

# Initial states:

InitElectr = np.asarray([CTH12 * CTH13, STH12 * CTH13, STH13 * np.exp(-1j * delta)])
InitMuon = np.asarray([-STH12 * CTH23 - CTH12 * STH23 * STH13 * np.exp(-1j * delta), \
	CTH12 * CTH23 - STH12 * STH23 * STH13 * np.exp(-1j * delta), STH23 * CTH13 ])
InitTaon = np.asarray([STH12 * STH23 - CTH12 * CTH23 * STH13 * np.exp(-1j * delta), \
	-CTH12 * STH23 - STH12 * CTH23 * STH13 * np.exp(-1j * delta), CTH23 * CTH13])
#                                           HAMILTONIAN

# a) Kinetic part of Hamiltonian (in the mass basis)

Kinetic = [[0 for i in range(0,3)] for j in range(0,3)] # empty 3-dimensional matrix

Kinetic[0][0] = m21
Kinetic[1][1] = m31

Kinetic = np.asarray(Kinetic)

factor = float(1) / float(2 * En)
Kinetic = factor * np.asarray(Kinetic)
KineticF = mulmul3(Mixing,Kinetic,np.conj(np.transpose(Mixing)))

# b) Potential part of the Hamiltonian (in the flavour basis)

Potential = [[0 for i in range(0,3)] for j in range(0,3)] # empty 3-dimensional matrix


# 												OBSERVABLES

# Parameters
 
BigCos = np.cos(2 * np.pi / 5)
LittleCos = np.cos(np.pi / 5)
BigSin = np.sin(2 * np.pi / 5)
LittleSin = np.sin(np.pi / 5)

# Normalization constant

N = float(1) / float(np.sqrt(1 + LittleCos))

# Vectors

v1 = N * np.asarray([np.sqrt(LittleCos),1,0])
v2 = N * np.asarray([np.sqrt(LittleCos),-LittleCos,LittleSin])
v3 = N * np.asarray([np.sqrt(LittleCos),BigCos,-BigSin])
v4 = N * np.asarray([np.sqrt(LittleCos),BigCos, BigSin])
v5 = N * np.asarray([np.sqrt(LittleCos),-LittleCos,-LittleSin])

# Observables

# Rotation

VectorSimple = np. asarray([1,0,0])

# The cross product of vectors - the value is the sin of the rotation angle

crossProduct = np.cross(InitElectr, VectorSimple)

# The coordinates of the cross product

w1 = crossProduct[0]
w2 = crossProduct[1]
w3 = crossProduct[2]

sinAlfa = vectorLength(crossProduct)

# The dot product of vectors - the value is the cos of the rotation angle

cosAlfa = np.dot(InitElectr, VectorSimple)

# Creation of the Rotation Matrix 

crossProductMatrix = np.asarray([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
rotationMatrix = np.identity(3) + crossProductMatrix + mulmul(crossProductMatrix, crossProductMatrix) * 1/(1 + cosAlfa)
rotationMatrixNew = np.linalg.inv(rotationMatrix)

# New vectors

v1New = mulvec(rotationMatrixNew, v1)
v2New = mulvec(rotationMatrixNew, v2)
v3New = mulvec(rotationMatrixNew, v3)
v4New = mulvec(rotationMatrixNew, v4)
v5New = mulvec(rotationMatrixNew, v5)

# Observables

A1 = -2 * np.outer(v1New,v1New) + np.identity(3)
A2 = -2 * np.outer(v2New,v2New) + np.identity(3)
A3 = -2 * np.outer(v3New,v3New) + np.identity(3)
A4 = -2 * np.outer(v4New,v4New) + np.identity(3)
A5 = -2 * np.outer(v5New,v5New) + np.identity(3)

#                                   UNITARY EVOLUTION

start = time.time()
Evolution = []
h = 0.001
for j in range(0,11):
	Potential[0][0] = 0.1 * j
	H = KineticF + Potential
	Evolution1 = []
	Czas = []
	for i in range(0,20000):
		t = h * i
		Czas.append(t)
		wykladnik = -1j * t * np.asarray(H)
		Unitary = SLA.expm(wykladnik)
		NewState = mulvec(Unitary,InitElectr)
		Density = outerpr(NewState,NewState)
		ExpW1 = np.trace(mulmul3(Density,A1,A2))
		ExpW2 = np.trace(mulmul3(Density,A2,A3))
		ExpW3 = np.trace(mulmul3(Density,A3,A4))
		ExpW4 = np.trace(mulmul3(Density,A4,A5))
		ExpW5 = np.trace(mulmul3(Density,A5,A1))
		suma = ExpW1 + ExpW2 + ExpW3 + ExpW4 + ExpW5
		Evolution1.append(suma)
	Evolution.append(Evolution1)

stop = time.time()
czas = stop - start
print("czas = {}".format(czas))

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

limit = 400

fig, ax1 = plt.subplots()
ax1.plot(Czas,Evolution[0], linewidth = 2, label = "$\mathbf{V_{CC}}=0$")
ax1.plot(Czas,Evolution[1],linewidth = 2,label = "$\mathbf{V_{CC}}=0.1$")
ax1.plot(Czas,Evolution[3],linewidth = 2,label = "$\mathbf{V_{CC}}=0.3$")
ax1.plot(Czas,Evolution[5],linewidth = 2,label = "$\mathbf{V_{CC}}=0.5$")
ax1.plot(Czas,Evolution[10],linewidth = 2,label = "$\mathbf{V_{CC}}=1$")
ax1.axhline(y = -3, color = 'black', linewidth = 3)
ax1.set_xlabel('$\mathbf{t}$', fontsize = 11)
ax1.set_ylabel(r'$\mathbf{\Delta}(t)$', fontsize = 11, va = 'baseline', labelpad = 10)
ax1.legend( prop=fontP, loc = 'upper right', frameon = True, fancybox = True, edgecolor = 'black')
ax1.set_ylim(-4, 1.5)


left, bottom, width, height = [0.25, 0.62, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot(Czas[:limit],Evolution[0][:limit], linewidth = 2, label = "$\mathbf{V_{CC}}=0$")
ax2.plot(Czas[:limit],Evolution[1][:limit],linewidth = 2,label = "$\mathbf{V_{CC}}=0.1$")
ax2.plot(Czas[:limit],Evolution[3][:limit],linewidth = 2,label = "$\mathbf{V_{CC}}=0.3$")
ax2.plot(Czas[:limit],Evolution[5][:limit],linewidth = 2,label = "$\mathbf{V_{CC}}=0.5$")
ax2.plot(Czas[:limit],Evolution[10][:limit],linewidth = 2,label = "$\mathbf{V_{CC}}=1$")


plt.show()



