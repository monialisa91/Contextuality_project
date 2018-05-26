import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import time
import matplotlib.pyplot as plt
import matplotlib

def mulvec(a,b):
	return np.einsum('ik,k->i',a,b)

def mulmul(a,b):
	return np.einsum ('ik,kj->ij',a,b)

def mulmul3(a,b,c):
	return np.einsum('ik,kj,jm->im',a,b,c)

# Inner Product for Complex Vectors (numpy provides only for the real ones)

def dotpr(v1,v2):
	v2 = np.conjugate(v2)
	return np.inner(v1,v2)

def outerpr(v1,v2):
	v1 = np.conjugate(v1)
	return np.outer(v1,v2)


#                                           PARAMETERS

# Mixing angles (sth- sinus 2*theta, cth- cosinus 2*theta)

sth13 = np.sqrt(0.0219)
sth23 = np.sqrt(0.304)
sth12 = np.sqrt(0.5)

cth13 = np.sqrt(1 - sth13 ** 2)
cth23 = np.sqrt(1 - sth23 ** 2)
cth12= np.sqrt(1 - sth12 ** 2)

STH13 = np.sqrt((1 - cth13) / 2)
STH23 = np.sqrt((1 - cth23) / 2)
STH12 = np.sqrt((1 - cth12) / 2)

CTH13 = np.sqrt((1 + cth13) / 2)
CTH23 = np.sqrt((1 + cth23) / 2)
CTH12 = np.sqrt((1 + cth12) / 2)

print("STH13:{}".format(STH13))
print("STH23:{}".format(STH23))
print("STH12:{}".format(STH12))

print("CTH13:{}".format(CTH13))
print("CTH23:{}".format(CTH23))
print("CTH12:{}".format(CTH12))


# The mass squares

m21 = 7.53 * 10
m32 = 2.44 * 10 ** (3)
m31 = m21 + m32
phases = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi, 3 * np.pi/2]


# The remaining parameters

En = 10 ** 9 # energy of the neutrino
VCC = 0.1 # charge-current potential
h = 0.01
delta = 0
factor = float(1) / float(2 * En)


#                                           MIXING MATRIX


Mixing = [[0 for i in range(0,3)] for j in range(0,3)] # empty 3-dimensional matrix

Mixing[0][0] = CTH12 * CTH13
Mixing[0][1] = STH12 * CTH13
Mixing[2][2] = CTH13 * CTH23

#                                           HAMILTONIAN

# a) Kinetic part of Hamiltonian (in the mass basis)

Kinetic = [[0 for i in range(0,3)] for j in range(0,3)] # empty 3-dimensional matrix

Kinetic[0][0] = m21
Kinetic[1][1] = m31

Kinetic = np.asarray(Kinetic)
Kinetic = factor * np.asarray(Kinetic)

# b) Potential part of the Hamiltonian (in the flavour basis)

Potential = [[0 for i in range(0,3)] for j in range(0,3)] # empty 3-dimensional matrix

Potential[0][0] = VCC

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

A1 = -2 * np.outer(v1,v1) + np.identity(3)
A2 = -2 * np.outer(v2,v2) + np.identity(3)
A3 = -2 * np.outer(v3,v3) + np.identity(3)
A4 = -2 * np.outer(v4,v4) + np.identity(3)
A5 = -2 * np.outer(v5,v5) + np.identity(3)

#                                   UNITARY EVOLUTION

# Initial states:

start = time.time()
Evolution = []

t = 0.5

for j in range(len(phases)):
	delta = phases[j]
	print("faza:{}".format(delta))
	InitElectr = np.asarray([CTH12 * CTH13, STH12 * CTH13, STH13 * np.exp(-1j * delta)])
	InitMuon = np.asarray([-STH12 * CTH23 - CTH12 * STH23 * STH13 * np.exp(-1j * delta), \
	CTH12 * CTH23 - STH12 * STH23 * STH13 * np.exp(-1j * delta), STH23 * CTH13 ])
	InitTaon = np.asarray([STH12 * STH23 - CTH12 * CTH23 * STH13 * np.exp(-1j * delta), \
	-CTH12 * STH23 - STH12 * CTH23 * STH13 * np.exp(-1j * delta), CTH23 * CTH13])
	Mixing[0][2] = STH13 * np.exp(-1j * delta)

	Mixing[1][0] =-STH12 * CTH23 - CTH12 * STH23 * STH13 * np.exp(-1j * delta)
	Mixing[1][1] = CTH12 * CTH23 - STH12 * STH23 * STH13 * np.exp(-1j * delta)
	Mixing[1][2] = STH23 * CTH13

	Mixing[2][0] = STH12 * STH23 - CTH12 * CTH23 * STH13 * np.exp(-1j * delta)
	Mixing[2][1] = -CTH12 * STH23 - STH12 * CTH23 * STH13 * np.exp(-1j * delta)

	KineticF = mulmul3(Mixing,Kinetic,np.conj(np.transpose(Mixing)))
	Evolution1 = []
	Potentials = []
	for i in range(100):
		H = KineticF + Potential
		Potentials.append(VCC)
		wykladnik = -1j * t * np.asarray(H)
		Unitary = SLA.expm(wykladnik)
		NewState = mulvec(Unitary,InitTaon)
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



plt.plot(Potentials,Evolution[0], linewidth = 2, label = "$\phi=0$")
plt.plot(Potentials,Evolution[1],'g-',linewidth = 2,label = "$\phi = \pi/6$")
plt.plot(Potentials,Evolution[2],'r-',linewidth = 2,label = "$\phi = \pi/4$")
plt.plot(Potentials,Evolution[3],'m-',linewidth = 2,label = "$\phi = \pi/3$")
plt.plot(Potentials,Evolution[4],'y-',linewidth = 2,label = "$\phi = \pi/2$")
plt.plot(Potentials,Evolution[5],'b-',linewidth = 2,label = "$\phi = \pi$")
plt.plot(Potentials,Evolution[6],'c-',linewidth = 2,label = "$\phi = 3\pi/2$")
#plt.axhline(y = -3, color = 'black', linewidth = 3, hold = None)
ax1 = plt.subplot()
ax1.set_xlabel('$V_{CC}$');
ax1.set_ylabel('$\Delta$');
plt.legend()
plt.show()